import requests
from bs4 import BeautifulSoup
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from langchain_core.tools import tool
from readability.readability import Document


async def creat_web_search_tool():
    api_key = os.getenv("API_KEY")
    mcp_config = {
        "web_search": {
            "transport": "streamable-http",
            "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
            "headers": {
                "Authorization": f"Bearer {api_key}"
            }
        }
    }
    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools()


@tool
def visit_url(url: str, encoding: str = "utf-8", max_length: int = 4000) -> str:
    """
    该工具用于浏览网页，读取网页内容返回文本信息
    :param url: 目标网页地址
    :param encoding: 网页编码
    :param max_length: 返回文本的最大长度
    :return:
        网页的文本内容
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; my-bot/1.0)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"Error: 打开网页失败{url}. Exp: {e}"
    try:
        doc = Document(response.text)
        main_content = doc.summary()
    except Exception as e:
        main_content = response.text

    main_text = BeautifulSoup(main_content, "html.parser").get_text(separator="\n", strip=True)

    return main_text if main_text else "Error: 提取网页失败"







@tool
def visit_url(url: str, encoding: str = None, max_length: int = 4000) -> str:
    """
    使用这个工具来浏览一个网页的详细内容，并返回解析后的文本内容。

    当你已经从搜索结果中看到某个网页可能包含详细信息，需要深入阅读时使用此工具。
    适合用于获取权威来源的详细内容、技术文档、新闻报道全文等。

    Args:
        url: 要访问的网页URL
        encoding: 网页编码（可选），如果不指定则自动检测
        max_length: 返回文本的最大长度（默认4000字符）

    Returns:
        str: 网页的主要文本内容（已清理HTML标签）
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; my-bot/1.0)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
    except Exception as e:
        return f"Error: Failed to fetch URL. Exception: {e}"

    if response.status_code != 200:
        return f"Error: Received status code {response.status_code}"

    # 处理编码
    if encoding:
        response.encoding = encoding
    else:
        response.encoding = response.apparent_encoding or response.encoding

    # 使用 readability 提取正文内容
    try:
        doc = Document(response.text)
        main_content = doc.summary()
    except Exception:
        main_content = response.text

    # 使用 BeautifulSoup 清理 HTML，提取纯文本
    main_text = BeautifulSoup(main_content, "html.parser").get_text(separator="\n", strip=True)

    # 限制长度
    if len(main_text) > max_length:
        main_text = main_text[:max_length] + "\n...（内容已截断）"

    return main_text if main_text else "Error: Failed to extract main content."
