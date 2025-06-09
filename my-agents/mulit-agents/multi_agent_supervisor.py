import getpass
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor


def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")




class MultiAgentSupervisor:
    def __init__(self):
        _set_env("OPENAI_API_KEY")
        _set_env("OPENAI_API_BASE")
        _set_env("TAVILY_API_KEY")
        self.research_agent = self.__get_research_agent__()
        self.math_agent = self.__get_math_agent__()
        self.supervisor_agent = self.__build_supervisor_agent__()

    def __build_supervisor_agent__(self):
        return create_supervisor(
            model=init_chat_model("openai:gpt-4.1"),
            agents=[self.research_agent, self.math_agent],
            prompt=(
                "You are a supervisor managing two agents:\n"
                "- a research agent. Assign research-related tasks to this agent\n"
                "- a math agent. Assign math-related tasks to this agent\n"
                "Assign work to one agent at a time, do not call agents in parallel.\n"
                "Do not do any work yourself."
            ),
            add_handoff_back_messages=True,
            output_mode="full_history",
        ).compile()
    def __get_research_agent__(self):
        web_search = TavilySearch(max_results=3)
        return create_react_agent(
            model="openai:gpt-4.1",
            tools=[web_search],
            prompt=(
                "You are a research agent.\n\n"
                "INSTRUCTIONS:\n"
                "- Assist ONLY with research-related tasks, DO NOT do any math\n"
                "- After you're done with your tasks, respond to the supervisor directly\n"
                "- Respond ONLY with the results of your work, do NOT include ANY other text."
            ),
            name="research_agent",
        )
    def __get_math_agent__(self):
        def add(a: float, b: float):
            """Add two numbers."""
            return a + b

        def multiply(a: float, b: float):
            """Multiply two numbers."""
            return a * b

        def divide(a: float, b: float):
            """Divide two numbers."""
            return a / b

        return create_react_agent(
            model="openai:gpt-4.1",
            tools=[add, multiply, divide],
            prompt=(
                "You are a math agent.\n\n"
                "INSTRUCTIONS:\n"
                "- Assist ONLY with math-related tasks\n"
                "- After you're done with your tasks, respond to the supervisor directly\n"
                "- Respond ONLY with the results of your work, do NOT include ANY other text."
            ),
            name="math_agent",
        )

    from langchain_core.messages import convert_to_messages

    def pretty_print_message(self,message, indent=False):
        pretty_message = message.pretty_repr(html=True)
        if not indent:
            print(pretty_message)
            return

        indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
        print(indented)

    def pretty_print_messages(self,update, last_message=False):
        is_subgraph = False
        if isinstance(update, tuple):
            ns, update = update
            # skip parent graph updates in the printouts
            if len(ns) == 0:
                return

            graph_id = ns[-1].split(":")[0]
            print(f"Update from subgraph {graph_id}:")
            print("\n")
            is_subgraph = True

        for node_name, node_update in update.items():
            update_label = f"Update from node {node_name}:"
            if is_subgraph:
                update_label = "\t" + update_label

            print(update_label)
            print("\n")

            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]

            for m in messages:
                self.pretty_print_message(m, indent=is_subgraph)
            print("\n")
    def execute(self,query:str):
        for chunk in self.supervisor.stream(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query,
                        }
                    ]
                },
        ):
            self.pretty_print_messages(chunk, last_message=True)


if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["OPENAI_API_BASE"] = ""
    os.environ["TAVILY_API_KEY"] = ""
    multi = MultiAgentSupervisor()
    multi.execute("find US and New York state GDP in 2024. what % of US GDP was New York state?")