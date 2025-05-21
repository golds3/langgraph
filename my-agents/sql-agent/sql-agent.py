"""
an agent to select sql with human message
"""
import getpass
import os
import uuid
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode

from libs.langgraph.langgraph.constants import START


def _set_env(key:str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


class SqlAgent:

    def __init__(self,custom:bool=False):
        """
        Args:
            custom: 是否自定义构建agent,默认使用prebuilt构建
        """
        _set_env("OPENAI_API_KEY")
        _set_env("OPENAI_API_BASE")
        self.llm = init_chat_model("openai:gpt-4.1-nano")
        self.db = SQLDatabase.from_uri("mysql+pymysql://root:root@192.168.196.195:3307/nfturbo?charset=utf8")
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()
        if  custom:
            self.agent = self.__create_agent_custom__()
        else:
            self.agent = self.__create_agent_prebuilt__()

    def __create_agent_custom__(self):
        """
        定制要求agent按照一下要求执行
        1.查看数据库有多少张表
        2.查看表结构schema
        3.生成查询sql
        4.检查sql正确性
        """
        get_schema_tool = next(tool for tool in self.tools if tool.name == "sql_db_schema")

        run_query_tool = next(tool for tool in self.tools if tool.name == "sql_db_query")
        def list_tables(state: MessagesState):
            tool_call = {
                "name": "sql_db_list_tables",
                "args": {},
                "id": str(uuid.uuid4()),
                "type": "tool_call",
            }
            tool_call_message = AIMessage(content="", tool_calls=[tool_call])

            list_tables_tool = next(tool for tool in self.tools if tool.name == "sql_db_list_tables")
            tool_message = list_tables_tool.invoke(tool_call)
            response = AIMessage(f"Available tables: {tool_message.content}")

            return {"messages": [tool_call_message, tool_message, response]}

        def call_get_schema(state: MessagesState):
            # Note that LangChain enforces that all models accept `tool_choice="any"`
            # as well as `tool_choice=<string name of tool>`.
            llm_with_tools = self.llm.bind_tools([get_schema_tool], tool_choice="any")
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        def generate_query(state: MessagesState):
            generate_query_system_prompt = """
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {dialect} query to run,
            then look at the results of the query and return the answer. Unless the user
            specifies a specific number of examples they wish to obtain, always limit your
            query to at most {top_k} results.

            You can order the results by a relevant column to return the most interesting
            examples in the database. Never query for all the columns from a specific table,
            only ask for the relevant columns given the question.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
            """.format(
                dialect=self.db.dialect,
                top_k=5,
            )
            system_message = {
                "role": "system",
                "content": generate_query_system_prompt,
            }
            # We do not force a tool call here, to allow the model to
            # respond naturally when it obtains the solution.
            llm_with_tools = self.llm.bind_tools([run_query_tool])
            response = llm_with_tools.invoke([system_message] + state["messages"])

            return {"messages": [response]}

        def check_query(state: MessagesState):
            check_query_system_prompt = """
            You are a SQL expert with a strong attention to detail.
            Double check the {dialect} query for common mistakes, including:
            - Using NOT IN with NULL values
            - Using UNION when UNION ALL should have been used
            - Using BETWEEN for exclusive ranges
            - Data type mismatch in predicates
            - Properly quoting identifiers
            - Using the correct number of arguments for functions
            - Casting to the correct data type
            - Using the proper columns for joins

            If there are any of the above mistakes, rewrite the query. If there are no mistakes,
            just reproduce the original query.

            You will call the appropriate tool to execute the query after running this check.
            """.format(dialect=self.db.dialect)
            system_message = {
                "role": "system",
                "content": check_query_system_prompt,
            }

            # Generate an artificial user message to check
            tool_call = state["messages"][-1].tool_calls[0]
            user_message = {"role": "user", "content": tool_call["args"]["query"]}
            llm_with_tools = self.llm.bind_tools([run_query_tool], tool_choice="any")
            response = llm_with_tools.invoke([system_message, user_message])
            response.id = state["messages"][-1].id

            return {"messages": [response]}

        # build graph
        def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
            messages = state["messages"]
            last_message = messages[-1]
            if not last_message.tool_calls:
                return END
            else:
                return "check_query"

        get_schema_node = ToolNode([get_schema_tool], name="get_schema")
        run_query_node = ToolNode([run_query_tool], name="run_query")
        builder = StateGraph(MessagesState)
        builder.add_node(list_tables)
        builder.add_node(call_get_schema)
        builder.add_node(get_schema_node, "get_schema")
        builder.add_node(generate_query)
        builder.add_node(check_query)
        builder.add_node(run_query_node, "run_query")

        builder.add_edge(START, "list_tables")
        builder.add_edge("list_tables", "call_get_schema")
        builder.add_edge("call_get_schema", "get_schema")
        builder.add_edge("get_schema", "generate_query")
        builder.add_conditional_edges(
            "generate_query",
            should_continue,
        )
        builder.add_edge("check_query", "run_query")
        builder.add_edge("run_query", "generate_query")

        return builder.compile()
    def __create_agent_prebuilt__(self):
        system_prompt = """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        Then you should query the schema of the most relevant tables.
        """.format(
            dialect=self.db.dialect,
            top_k=5,
        )

        return create_react_agent(
            self.llm,
            self.tools,
            prompt=system_prompt,
        )

    def query(self, question:str):
        for step in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="values",
        ):
            step["messages"][-1].pretty_print()


if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["OPENAI_API_BASE"] = ""
    # sql_agent = SqlAgent()
    # # 查询哪个用户购买的藏品数量最多
    # sql_agent.query("有几个用户角色是CUSTOMER的，他们的手机号是多少")

    sql_agent_custom = SqlAgent(custom=True)
    sql_agent_custom.query("最近半年，哪个用户做的交易数量最多，交易金额是多少")

