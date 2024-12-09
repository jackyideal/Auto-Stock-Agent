# from openai import OpenAI
#
# client = OpenAI(
#     api_key="sk-q5FSdHU6M1SE3bMkMqzZRClqVCSfJdCqI2ueasHh36sLqbIB",
#     base_url="https://api.moonshot.cn/v1",
# )
#
# completion = client.chat.completions.create(
#     model="moonshot-v1-128k",
#     messages=[
#         {"role": "system",
#          "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
#         {"role": "user", "content": "青松建化最新股价是多少？"}
#     ],
#     temperature=0.3,
# )
#
# print(completion.choices[0].message.content)







from typing import *

import os
import json

from openai import OpenAI
from openai.types.chat.chat_completion import Choice

client = OpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key="",
)


def search_impl(arguments: Dict[str, Any]) -> Any:
    """
    在使用 Moonshot AI 提供的 search 工具的场合，只需要原封不动返回 arguments 即可，
    不需要额外的处理逻辑。

    但如果你想使用其他模型，并保留联网搜索的功能，那你只需要修改这里的实现（例如调用搜索
    和获取网页内容等），函数签名不变，依然是 work 的。

    这最大程度保证了兼容性，允许你在不同的模型间切换，并且不需要对代码有破坏性的修改。
    """
    return arguments


def chat(messages) -> Choice:
    completion = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=messages,
        temperature=0.3,
        tools=[
            {
                "type": "builtin_function",
                "function": {
                    "name": "$web_search",
                },
            }
        ]
    )
    return completion.choices[0]


def main():
    messages = [
        {"role": "system", "content": "你是 Kimi。"},
    ]


    messages.append({
        "role": "user",
        "content": "请搜索贵州茅台相关的新闻，并将相关新闻网站的链接也展示出来，谢谢！同时告诉我贵州茅台的最近新闻有什么，谢谢！"
    })

    finish_reason = None
    while finish_reason is None or finish_reason == "tool_calls":
        choice = chat(messages)
        finish_reason = choice.finish_reason
        if finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                tool_call_name = tool_call.function.name
                tool_call_arguments = json.loads(
                    tool_call.function.arguments)
                if tool_call_name == "$web_search":
                    tool_result = search_impl(tool_call_arguments)
                else:
                    tool_result = f"Error: unable to find tool by name '{tool_call_name}'"


                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": json.dumps(tool_result),

                })

    print(choice.message.content)


if __name__ == '__main__':
    main()