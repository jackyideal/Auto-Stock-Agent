# import asyncio
# from autogen_agentchat.agents import AssistantAgent
# from autogen_agentchat.task import Console, TextMentionTermination
# from autogen_agentchat.teams import RoundRobinGroupChat
# from autogen_ext.models import OpenAIChatCompletionClient
#
# # Define a tool
# async def get_weather(city: str) -> str:
#     return f"The weather in {city} is 73 degrees and Sunny."
#
# async def main() -> None:
#     # Define an agent
#     weather_agent = AssistantAgent(
#         name="weather_agent",
#         model_client=OpenAIChatCompletionClient(
#             model="gpt-3.5-turbo",
#             api_key = "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"),
#         tools=[get_weather],
#     )
#
#
#     # Define termination condition
#     termination = TextMentionTermination("TERMINATE")
#
#     # Define a team
#     agent_team = RoundRobinGroupChat([weather_agent], termination_condition=termination)
#
#     # Run the team and stream messages to the console
#     stream = agent_team.run_stream(task="去东莞玩好吗？")
#     # for message in stream:
#     #     print("-------message------", message)
#     print("stream",stream)
#
#     await Console(stream)
#
# asyncio.run(main())



import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import Console, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import OpenAIChatCompletionClient

# Define a tool
async def get_weather(city: str) -> str:
    return f"The weather in {city} is 73 degrees and Sunny."

async def main() -> None:
    # Define an agent
    weather_agent = AssistantAgent(
        name="weather_agent",
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
            api_key="",
        ),
        tools=[get_weather],
    )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")

    # Define a team
    agent_team = RoundRobinGroupChat([weather_agent], termination_condition=termination)

    # Run the team and stream messages to the console
    stream = agent_team.run_stream(task="What is the weather in New York?")
    await Console(stream)

asyncio.run(main())