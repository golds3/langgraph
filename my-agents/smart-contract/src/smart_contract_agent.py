import os
from getpass import getpass

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from langgraph_supervisor import create_supervisor
from agents import SqlAgent, RagAgent


def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


# 加载 .env 文件
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_API_BASE"] = openai_api_base

# agent 协调者
# sql agent
sql_agent = SqlAgent(custom=True).agent
# rag agent
rag_agent = RagAgent().agent

supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4.1-nano"),
    agents=[sql_agent, rag_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a sql agent. Used when you need to query specific data from a database\n"
        "- a search agent. Used when it is necessary to retrieve information in documents and knowledge bases\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=False,
    output_mode="full_history",
).compile()


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
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
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


def execute(query: str):
    for chunk in supervisor.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            },
    ):
        pretty_print_messages(chunk, last_message=True)


execute("什么是票据监管？另外有哪些账号签约了票据监管")
