import json
import os
from typing import List

from dotenv import load_dotenv
from huggingface_hub.file_download import repo_folder_name
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from numpy.f2py.crackfortran import true_intent_list
from pydantic import BaseModel,Field
import gradio as gr


load_dotenv(r"../../.env")


def get_llm():
    api_key = os.getenv("API_KEY")
    assert api_key, ValueError("not find api key")
    return ChatOpenAI(
        model="qwen-turbo",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


system_prompt = """
     你是一个文本分析专家
     你的目标是：从用户提供的文本中提取实体以及实体间的关系。
     你需要按照要求完成以下任务
     1. 仔细的分析文本，提取文本中出现的主体
     2. 找出实体与实体之间的关系
     注意：
     * 不要输出重复的实体和实体间关系
     输出：
     只允许输出json格式数据，不要输出过多的内容
 """

prompt_template = ChatPromptTemplate([
    ("system", system_prompt),
    ("human", "{user_input}")
])


class Response(BaseModel):
    summary: str = Field(description="总结")
    entities: List[str] = Field(description="实体列表")
    relationship_list: List[List[str]] = Field(description="实体与实体间关系列表，每个item按照以下格式保存 [实体1, 关系， 实体2]")



def respond(message):
    llm = get_llm()
    llm = llm.with_structured_output(Response)
    prompt = prompt_template.format(user_input=message)
    response = llm.invoke(prompt)
    # 开始组织md文档
    entity_str = "## 实体列表\n"
    for entity in response.entities:
        entity_str += f"* {entity}\n"

    relationship_str = "## 实体关系\n"
    for relationship in response.relationship_list:
        relationship_str += f"* {"-->".join(relationship)}\n"

    md_output = (f"# 总结：\n"
                 f"{response.summary}\n"
                 f"{entity_str}\n"
                 f"{relationship_str}\n---\n"
                 f"```json\n"
                 f"{response.model_dump_json()}\n```")


    return md_output, "处理完成"

def main():

    with gr.Blocks(title="结构化输出") as demo:
        gr.Markdown("# 结构化提取")
        with gr.Row():
            with gr.Column(scale=2):
                tbox_user_input = gr.Textbox(label="输入文本", placeholder="输入您的文本", lines=10)
                btn_process = gr.Button("开始")
            with gr.Column(scale=3):
                tbox_status = gr.Text(label="状态", value="未开始", interactive=False)
                md_output  = gr.Markdown(label="分析结果", value="点击开始查看结果")

        btn_process.click(
            fn=respond,
            inputs=[tbox_user_input],
            outputs=[md_output, tbox_status]
        )
    demo.launch()



if __name__ == '__main__':
    main()
