import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import HumanMessage

load_dotenv(r"../../.env")


def get_llm():
    api_key = os.getenv("API_KEY")
    return ChatOpenAI(model="qwen-turbo",
                      api_key=api_key,
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


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


def main():
    app = build_chatbot()
    print("开始聊天")
    messages = []
    while True:
        try:
            user_input = input("\n你：").strip()
            if user_input.lower() in ["quite", "exit", "退出"]:
                print("再见")
                break
            if not user_input:
                continue

            messages.append(HumanMessage(content=user_input))
            print("AI: ", end="", flush=True)
            result = app.invoke({"messages": messages})
            ai_response = result["messages"][-1]
            print(ai_response.content)
            messages.append(ai_response)

        except KeyboardInterrupt:
            print("再见")
            break
        except Exception as e:
            print(f"error{str(e)}")
            break


if __name__ == '__main__':
    main()
