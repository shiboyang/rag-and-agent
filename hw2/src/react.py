import os

from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv(r"../../.env")


@tool
def convert_length(value: float, from_unit: str, to_unit: str) -> str:
    """
    长度单位换算工具：支持米(m)和英尺(ft)之间的转换。

    Args:
        value: 要转换的数值（必须是正数）
        from_unit: 原始单位，'m' 表示米，'ft' 表示英尺
        to_unit: 目标单位，'m' 表示米，'ft' 表示英尺

    Returns:
        换算结果的字符串描述，例如："5.0 米 = 16.40 英尺"

    Raises:
        ValueError: 当数值为负数或单位不支持时

    Example:
        >>> convert_length.invoke({"value": 5.0, "from_unit": "m", "to_unit": "ft"})
        "5.0 米 = 16.40 英尺"
        >>> convert_length.invoke({"value": 10.0, "from_unit": "ft", "to_unit": "m"})
        "10.0 英尺 = 3.05 米"
    """
    # 换算公式：1 米 = 3.28084 英尺
    factor = 3.28084
    if value < 0:
        raise ValueError(f"value must be > 0, get {value}")

    if from_unit.lower() == "m":
        return f"{value} 米 = {value * factor} 英尺"
    elif from_unit.lower() == "ft":
        return f"{value} 英尺 = {value / factor:.5f} 米"
    else:
        raise ValueError(f"only support unit m, ft")


@tool
def convert_weight(value: float, from_unit: str, to_unit: str) -> str:
    """
    重量单位换算工具：支持千克(kg)和磅(lb)之间的转换。

    Args:
        value: 要转换的数值（必须是正数）
        from_unit: 原始单位，'kg' 表示千克，'lb' 表示磅
        to_unit: 目标单位，'kg' 表示千克，'lb' 表示磅

    Returns:
        换算结果的字符串描述，例如："10.0 千克 = 22.05 磅"

    Example:
        >>> convert_weight.invoke({"value": 10.0, "from_unit": "kg", "to_unit": "lb"})
        "10.0 千克 = 22.05 磅"
    """
    # 换算公式：1 千克 = 2.20462 磅
    factor = 2.20462
    if value < 0:
        raise ValueError(f"value must be > 0, get {value}")

    if from_unit.lower() == "kg":
        return f"{value} 千克 = {value * factor} 磅"
    elif from_unit.lower() == "lb":
        return f"{value} 磅 = {value / factor:.5f} 千克"
    else:
        raise ValueError(f"only support unit kg, lb")


@tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    温度单位换算工具：支持摄氏度(C)和华氏度(F)之间的转换。

    Args:
        value: 要转换的温度数值
        from_unit: 原始单位，'c' 表示摄氏度，'f' 表示华氏度
        to_unit: 目标单位，'c' 表示摄氏度，'f' 表示华氏度

    Returns:
        换算结果的字符串描述，例如："30.0C = 86.0F"

    Example:
        >>> convert_temperature.invoke({"value": 30.0, "from_unit": "c", "to_unit": "f"})
        "30.0C = 86.0F"
        >>> convert_temperature.invoke({"value": 100.0, "from_unit": "f", "to_unit": "c"})
        "100.0F = 37.78C"
    """
    # 换算公式：F = C × 9/5 + 32
    #          C = (F - 32) × 5/9

    if from_unit.lower() == "c":
        return f"{value} 摄氏度 = {value * 9 / 5 + 32} 华氏度"
    elif from_unit.lower() == "lb":
        return f"{value} 华氏度 = {(value - 32) * 5 / 9} 摄氏度"
    else:
        raise ValueError(f"only support unit c, f")


def get_llm_with_tools():
    api_key = os.getenv("API_KEY")
    assert api_key
    llm = ChatOpenAI(
        model="qwen-turbo",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    tools = [convert_length, convert_weight, convert_temperature]
    llm = llm.bind_tools(tools)
    return llm


class AgentState(MessagesState):
    pass


def agent_node(state: AgentState):
    llm = get_llm_with_tools()
    system_prompt = """你是一个单位换算助手。
【职责】
帮助用户完成各种单位换算，包括长度、重量、温度。

【支持的工具】
1. convert_length: 长度换算（仅支持米和英尺）
2. convert_weight: 重量换算（仅支持千克和磅）
3. convert_temperature: 温度换算（仅支持摄氏度和华氏度）

【工作流程】
1. 识别用户想要换算的数值、原始单位和目标单位
2. 判断换算类型，调用相应的工具
3. 将工具返回的结果整理成友好的回答

【重要规则】
1. 必须使用工具获取结果，严禁自己计算或编造答案
2. 如果工具返回错误信息，直接将该错误信息友好地转述给用户
3. 如果用户没有提供完整信息，请礼貌地追问
4. 回答要清晰、准确，可以适当解释换算过程

【错误处理示例】
- 用户问："把5公里转换成英里"
  → 你应该调用 convert_length 工具，工具会返回错误提示
  → 你将错误提示转述为："抱歉，目前只支持米(m)和英尺(ft)之间的换算。"

- 用户问："帮我换算一下"
  → 你应该回复："请问您想换算什么内容呢？请提供具体的数值、原始单位和目标单位。"
  """
    message_with_system = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(message_with_system)
    return {"messages": [response]}


def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_result = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            print(f"调用工具{tool_name}, 参数：{tool_args}")

            if tool_name == "convert_length":
                result = convert_length.invoke(tool_args)
            elif tool_name == "convert_weight":
                result = convert_weight.invoke(tool_args)
            elif tool_name == "convert_temperature":
                result = convert_temperature.invoke(tool_args)
            else:
                result = f"未知的工具: {tool_name}"
            tool_result.append(
                ToolMessage(content=str(result), tool_call_id=tool_id)
            )
        return {"messages": tool_result}
    return {"messages": []}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def build_react_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def respond(message, history):
    print("message: ", message)
    print("history: ", history)
    messages = []
    for msg in history:
        if msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
    messages.append(HumanMessage(content=message))
    agent = build_react_agent()
    response = agent.invoke({"messages": messages})
    print(response)
    ai_message = response["messages"][-1].content

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ai_message}
    ]
    return "", new_history


def main():
    with gr.Blocks(title="React Agent") as demo:
        chat_bot = gr.Chatbot(height=500, placeholder="开始聊天")
        txt_msg = gr.Textbox(label="输入消息", placeholder="请输入内容")
        btn_submit = gr.Button("send", variant="primary")

        btn_submit.click(
            fn=respond,
            inputs=[txt_msg, chat_bot],
            outputs=[txt_msg, chat_bot]
        )
        demo.launch(
            share=False,
            inbrowser=False,
            prevent_thread_lock=False,  # 主线程阻塞，调试器不会退出
            debug=True  # gradio自带调试日志，打印接口请求)
        )


if __name__ == '__main__':
    main()
