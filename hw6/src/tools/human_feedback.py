from langchain_core.tools import tool


@tool
def get_more_information(question: str) -> str:
    """
    该工具用于向用户提问，以获取更都有效信息。当你认为有必要主动询问用户时，使用这个工具。
    :param question: 像用户提问的问题
    :return:
        用户的回答
    """
    try:
        user_input = input(f"问题：{question}\n[User]: ")
        return user_input.strip() if user_input.strip() else "用户未提供任何有效信息"
    except (EOFError, KeyboardInterrupt):
        return "用户中断了交互"

