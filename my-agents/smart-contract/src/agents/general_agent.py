import os
from enum import Enum
from getpass import getpass
from typing import Optional
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from rag_agent import RagAgent


# 加载 .env 文件
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_API_BASE"] = openai_api_base


from functools import lru_cache

# 缓存，避免重复生成
@lru_cache(maxsize=1)
def get_rag_agent() -> RagAgent:
    return RagAgent()
@tool(name_or_callable="rules_tool",description="查询某种类型合约的规则说明")
def rules_tool(contract_type: str) -> str:

    return """基础合约规则说明如下：
    
    1.如果合约的监管类型包含票据，那么不允许客户开通邮件通知功能
    2.如果客户开通了邮件通知功能,那么一定要维护至少两个邮箱地址，且不允许维护qq邮箱
    3.票据监管是指包含质押、贴现、背书等以票据作为凭据的交易监管
    4.如果要进行票据监管，必须要维护票据监管类型，如果不进行票据监管，则不能维护票据监管类型"""

# todo 接入rag_agent
# @tool(name_or_callable="rules_tool",description="查询某种类型合约的规则说明")
# def rules_tool(contract_type: str) -> str:
#     agent = get_rag_agent()
#     question = f"请提供关于 `{contract_type}` 类型合约的规则说明。"
#
#     chunks = []
#     stream = agent.agent.stream({"messages": [{"role": "user", "content": question}]})
#     for chunk in stream:
#         for node_update in chunk.values():
#             # 保险起见判断是不是字典
#             if isinstance(node_update, dict):
#                 messages = node_update.get("messages", [])
#                 if messages and isinstance(messages[-1], BaseMessage):
#                     if messages[-1].content:
#                         chunks.append(messages[-1].content)
#     return "\n".join(chunks) if chunks else "未查询到相关规则。"
class RegTypeEnum(str, Enum):
    deposit = "10"   # 存款监管
    bill = "01"      # 票据监管
    both = "11"      # 存款+票据监管
class Contract(BaseModel):
    acct_no: str = Field(description="合约签约账号")
    email_flag:int = Field(description="是否开通邮箱功能 0-不开通 1-开通")
    email: Optional[list[str]] = Field(description="the email address")
    reg_type:RegTypeEnum = Field("监管类型 '10'-存款监管 '01'-票据监管 '11'-存款+票据监管")
    bill_reg_type:Optional[str] = Field(description="票据监管类型 '000'-不签约票据监管 '100'-质押 '110'-质押+背书 '111'-质押+背书+贴现 '101'-质押+贴现 '011'-背书+贴现")

struct_output_agent = create_react_agent(
    model="gpt-4o-mini",
    tools=[rules_tool],
    response_format=Contract,
    prompt=(
        "You are a helpful assistant that generates example contracts in structured form.\n"
        "Always begin by calling the `rules_tool` tool to retrieve rules for the `contract_type='基础合约'`.\n"
        "Then infer field values according to the rules.\n"
        "If a field is optional and not inferable, you may omit it.\n"
        "Return a result that fits the `Contract` structure."
    )
)


for chunk in struct_output_agent.stream(
    {"message": [{"role": "user", "content": "生成一个基础合约,只进行票据监管，包含质押和背书"}]}
):
    print(chunk)
