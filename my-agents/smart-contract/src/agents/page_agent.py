import json
import os
from typing import Optional

import dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_API_BASE"] = openai_api_base


class Model(BaseModel):
    acct_no: str = Field(description="合约签约账号")
    email_flag: int = Field(description="是否开通邮箱功能 0-不开通 1-开通")
    email: Optional[list[str]] = Field(description="the email address")
    reg_type: str = Field("走款选项 '10'-存款监管 '01'-票据监管 '11'-存款+票据监管")
    bill_reg_type: Optional[str] = Field(
        description="票据监管类型 '000'-不签约票据监管 '100'-质押 '110'-质押+背书 '111'-质押+背书+贴现 '101'-质押+贴现 '011'-背书+贴现")

    def to_json_with_descriptions(self) -> str:
        # 创建一个包含字段名、值和描述的字典
        data_with_descriptions = {
            field_name: {
                "value": getattr(self, field_name),
                "description": field.description
            }
            for field_name, field in self.__fields__.items()
        }
        return json.dumps(data_with_descriptions, ensure_ascii=False, indent=4)
@tool(description="get json data")
def data_tool() -> str:
    m = Model(
        acct_no='dasad111',
        email_flag=1,  # 这里应该是整数而不是字符串
        email=["1234@qq.com"],  # 这里应该是列表
        reg_type='10',
        bill_reg_type='111'
    )
    return m.to_json_with_descriptions()


class HtmlModel(BaseModel):
    content: str = Field(description="html 文件的内容")

page_agent = create_react_agent(
    model="gpt-4o-mini",
    tools=[data_tool],
    response_format=HtmlModel,
    prompt=(
        "你是一个html页面生成助手，你必须先调用data_tool去获取json数据，然后把json数组渲染到html页面。"
        "页面样式要好看点，针对json字段，你要展示它的中文名称而不是它的值"
        "页面的标题是`合约样例`"
    )
)

for chunk in page_agent.stream(
        {
            "message": [
                {
                    "role": "user",
                    "content": "帮我生成一个页面"
                 }
            ]
        }
):
    print(chunk)

html_data = chunk["generate_structured_response"]["structured_response"].content
with open("general.html", "w", encoding="utf-8") as f:
    f.write(html_data)