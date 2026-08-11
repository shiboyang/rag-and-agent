from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")


@mcp.tool()
def get_weather(city: str) -> str:
    """
    根据城市查询天气情况
    :param city: 中文城市名
    :return: 天气信息
    """
    fake_weather_dict = {
        "巴黎": "晴 25摄氏度",
        "北京": "多云 30摄氏度",
        "上海": "雨 22摄氏度",
        "纽约": "雪 -5摄氏度"
    }
    return fake_weather_dict.get(city, f"未查询到{city}的天气数据")


if __name__ == '__main__':
    mcp.run()
