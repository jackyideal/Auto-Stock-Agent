

import base64
import time
import requests
import os


api_key = ""


def encode_image(image_path):
    """
    将图片编码为 Base64 格式
    :param image_path: 图片路径
    :return: Base64 编码后的字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_k_line(image_path):
    """
    分析股票 K 线图
    :param image_path: 图片路径
    :return: GPT-4 Vision 返回的分析结果
    """

    base64_image = encode_image(image_path)


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "你是一个优秀的数据分析师，请根据下方提供的股票K线图，对以下内容进行详细的技术分析：\n"
                            "1. **股价走势**：分析主要趋势（上涨、下跌、震荡）及其可能的变化。\n"
                            "2. **技术指标**：结合图中的MACD、KDJ、布林带等指标，判断当前市场状态（超买、超卖或趋势变化）。\n"
                            "3. **成交量分析**：结合成交量柱状图，判断市场资金动向及投资者情绪。\n"
                            "4. **综合分析与预测**：结合以上内容，对未来短期和中期的股价走势进行预判，并给出操作建议（如买入、持有或卖出）。\n"
                            "请在分析中提及具体的技术信号和指标变化，例如金叉、死叉、趋势线突破等，确保分析内容详实且逻辑清晰。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 2000,
    }


    while True:
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            time.sleep(1)
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            time.sleep(1)
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            time.sleep(1)
        except requests.exceptions.RequestException as err:
            print(f"An error occurred: {err}")
            time.sleep(1)


if __name__ == "__main__":

    image_path = '/autostock/600425.SH_kline.png'
    result = analyze_k_line(image_path)
    print("\n=== K线图分析结果 ===\n")
    print(result)
