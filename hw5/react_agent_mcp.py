import asyncio
import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(r"../.env")


def load_mcp_config():
    api_key = os.getenv("API_KEY")
    mcp_config = {
        "file_system": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/home/shiby/code/rag-and-agent"
            ]
        },
        "web_search": {
            "transport": "streamable-http",
            "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
            "headers": {
                "Authorization": f"Bearer {api_key}"
            }
        },
        "weather": {
            "transport": "stdio",
            "command": "/home/shiby/miniconda3/envs/langgraph/bin/python",
            "args": ["/home/shiby/code/rag-and-agent/hw5/weather_server.py"]
        }
    }
    return mcp_config


async def create_mcp_agent():
    mcp_config = load_mcp_config()
    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools()

    llm = ChatOpenAI(
        model="qwen-turbo",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
        api_key=os.getenv("API_KEY")
    )

    agent = create_agent(llm, tools)

    return agent


async def test_agent_query(question, agent):
    print(f"问题：{question}")
    result = await agent.ainvoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    answer = result["messages"][-1].content

    print(f"回答：{answer}")


async def main():
    mcp_agent = await create_mcp_agent()
    question = "看下这个/home/shiby/code/rag-and-agent目录中有哪些文件，然后查询一下Python LangGraph的介绍"
    await test_agent_query(question, mcp_agent)


if __name__ == '__main__':
    asyncio.run(main())
