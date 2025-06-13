import getpass
import os
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_core.messages import HumanMessage
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field


def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


class RagAgent:

    def __init__(self):
        _set_env("OPENAI_API_KEY")
        _set_env("OPENAI_API_BASE")
        self.retriever_tool = self.__prepare_context__()
        self.response_model = init_chat_model("openai:gpt-4.1-nano", temperature=0)
        self.agent = self.__build_agent_graph__()

    def __prepare_context__(self):
        """构建知识库"""
        docs_list = TextLoader(file_path="/Users/ham/Desktop/project/ai/langgraph/my-agents/smart-contract/sources/合约规则说明.txt").load()
        # docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)
        vectorstore = InMemoryVectorStore.from_documents(
            documents=doc_splits, embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()
        return create_retriever_tool(
            retriever,
            "contract_rules",
            "Search and return information about contract rules.",
        )


    def rag_query(self, query: str):
        for chunk in self.agent.stream(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query,
                        }
                    ]
                }
        ):
            for node, update in chunk.items():
                print("Update from node", node)
                update["messages"][-1].pretty_print()
                print("\n\n")

    class GradeDocuments(BaseModel):
        """Grade documents using a binary score for relevance check."""

        binary_score: str = Field(
            description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
        )
    def __build_agent_graph__(self):

        def generate_query_or_respond(state: MessagesState):
            """Call the model to generate a response based on the current state. Given
                the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
            """
            response = self.response_model.bind_tools([self.retriever_tool]).invoke(state["messages"])
            return {"messages": [response]}
        def grade_documents(
                state: MessagesState,
        ) -> Literal["generate_answer", "rewrite_question"]:
            grader_model = init_chat_model("openai:gpt-4.1-nano", temperature=0)
            GRADE_PROMPT = (
                "You are a grader assessing relevance of a retrieved document to a user question. \n "
                "Here is the retrieved document: \n\n {context} \n\n"
                "Here is the user question: {question} \n"
                "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
                "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
            )
            """Determine whether the retrieved documents are relevant to the question."""
            question = state["messages"][0].content
            context = state["messages"][-1].content

            prompt = GRADE_PROMPT.format(question=question, context=context)
            response = (
                grader_model
                .with_structured_output(self.GradeDocuments).invoke(
                    [{"role": "user", "content": prompt}]
                )
            )
            score = response.binary_score

            if score == "yes":
                return "generate_answer"
            else:
                return "rewrite_question"

        def rewrite_question(state: MessagesState):
            REWRITE_PROMPT = (
                "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
                "Here is the initial question:"
                "\n ------- \n"
                "{question}"
                "\n ------- \n"
                "Formulate an improved question:"
            )
            """Rewrite the original user question."""
            messages = state["messages"]
            question = messages[0].content
            prompt = REWRITE_PROMPT.format(question=question)
            response = self.response_model.invoke([{"role": "user", "content": prompt}])
            return {"messages": [HumanMessage(content=response.content)]}

        def generate_answer(state: MessagesState):
            GENERATE_PROMPT = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Use three sentences maximum and keep the answer concise.\n"
                "Question: {question} \n"
                "Context: {context}"
                "- After you're done with your tasks, respond to the supervisor directly\n"
                "- Respond ONLY with the results of your work, do NOT include ANY other text."
            )

            """Generate an answer."""
            question = state["messages"][0].content
            context = state["messages"][-1].content
            prompt = GENERATE_PROMPT.format(question=question, context=context)
            response = self.response_model.invoke([{"role": "user", "content": prompt}])
            return {"messages": [response]}

        workflow = StateGraph(MessagesState)

        # Define the nodes we will cycle between
        workflow.add_node(generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node(rewrite_question)
        workflow.add_node(generate_answer)

        workflow.add_edge(START, "generate_query_or_respond")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            # Assess LLM decision (call `retriever_tool` tool or respond to the user)
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        # Edges taken after the `action` node is called.
        workflow.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            grade_documents,
        )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        # Compile
        return workflow.compile(name='rag_agent')


if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = "sk-FS0MXRcux9jhT9Mx84a92k2R8gPA3AMldHG5oRfn9HrXQl0O"
    os.environ["OPENAI_API_BASE"] = "https://www.DMXapi.com/v1/"
    rag_agent = RagAgent()
    input = "如果账号要开通邮箱功能，有什么限制"
    rag_agent.rag_query(input)
