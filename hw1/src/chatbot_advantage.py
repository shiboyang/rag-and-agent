import os
from fcntl import FASYNC

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr

load_dotenv(r"../../.env")


def get_llm():
    api_key = os.getenv("API_KEY")
    return ChatOpenAI(model="qwen-turbo",
                      api_key=api_key,
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      streaming=True)


def chat_node(state: MessagesState):
    llm = get_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_chatbot():
    workflow = StateGraph(MessagesState)
    workflow.add_node("chat", chat_node)
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", "__end__")
    return workflow.compile()


async def respond_stream(message, history, reset=False):
    if reset:
        yield "", []
        return
    if not message:
        yield "", history
        return

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]
    yield "", new_history

    try:
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            if msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=message))

        app = build_chatbot()
        full_response = ""

        async for event in app.astream_events({"messages": messages}, version="v1"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    full_response += chunk.content
                    new_history[-1]["content"] = full_response
                    yield "", new_history

    except Exception as e:
        new_history[-1]["content"] = f"error{str(e)}"
        yield "", new_history


def main():
    with gr.Blocks(title="Qwen Chatbot") as demo:
        gr.Markdown("# Qwen Chatbot")

        chatbot = gr.Chatbot(height=500, placeholder="开始聊天")
        msg = gr.Textbox(label="输入消息", placeholder="在这里输入你的你的问题", lines=1)
        with gr.Row():
            reset_btn = gr.Button("reset", variant="secondary")
            submit_btn = gr.Button("send", variant="primary")

        gr.Markdown("---")
        gr.Markdown("按下Enter发送消息")

        submit_btn.click(
            fn=respond_stream,
            inputs=[msg, chatbot, gr.State(False)],
            outputs=[msg, chatbot]
        )
        msg.submit(
            fn=respond_stream,
            inputs=[msg, chatbot, gr.State(False)],
            outputs=[msg, chatbot]
        )

        reset_btn.click(
            fn=respond_stream,
            inputs=[gr.State(""),chatbot,gr.State(True)],
            outputs=[msg,chatbot]
        )


        demo.launch(
            server_name="0.0.0.0",
            server_port=8888,
            share=False,
            show_error=True,
            theme=gr.themes.Soft()
        )


if __name__ == '__main__':
    main()
