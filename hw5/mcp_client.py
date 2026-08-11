import asyncio
import json
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def load_mcp_configs():
    api_key = os.getenv("API_KEY")

    mcp_configs = {
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
        "weather":{
            "transport":"stdio",
            "command":"/home/shiby/miniconda3/envs/langgraph/bin/python",
            "args":["/home/shiby/code/rag-and-agent/hw5/weather_server.py"]
        }
    }

    return mcp_configs


async def main():
    mcp_config = load_mcp_configs()
    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools()
    print(f"{[tool.name for tool in tools]}")

    for tool in tools:
        print(f"工具名称：{tool.name}")
        print(f"工具描述: {tool.description}")

        if hasattr(tool, "args_schema"):
            if isinstance(tool.args_schema, dict):
                schema = tool.args_schema
            elif hasattr(tool.args_schema, "schema"):
                schema = tool.args_schema.model_json_schema()
            else:
                schema = str(tool.args_schema)

            if isinstance(schema, dict):
                print(f"参数要求：{json.dumps(schema, indent=2, ensure_ascii=False)}")


if __name__ == '__main__':
    asyncio.run(main())
