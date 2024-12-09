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


def kimi_web_search(task):

    messages = [
        {"role": "system", "content": "你是 Kimi。"},
    ]

    # 初始提问
    messages.append({
        "role": "user",
        "content": f"{task} + 请搜索这个用户问题的新闻，并将相关新闻网站的链接也展示出来，谢谢！同时请总结这些链接的信息，提供最近新闻的总结，字数可以尽可能的多，内容详细。谢谢！"
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
    return choice.message.content


