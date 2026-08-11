import asyncio
import os
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv(r"../.env")


async def main():
    here = Path(__file__).resolve().parent
    server_py = str(here / "weather_server.py")

    client = MultiServerMCPClient({
        "weather": {
            "transport": "stdio",
            "command": "/home/shiby/miniconda3/envs/langgraph/bin/python",
            "args": [server_py]
        }
    })

    tools = await client.get_tools()
    print(f"可使用的工具： {[tool.name for tool in tools]}")

    llm = ChatOpenAI(
        model="qwen-turbo",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
        api_key=os.getenv("API_KEY")
    )

    agent = create_agent(llm, tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "巴黎的天气怎么样"}]
    })
    print(result["messages"][-1].content)


if __name__ == '__main__':
    asyncio.run(main())
