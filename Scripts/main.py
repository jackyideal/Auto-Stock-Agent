import gradio as gr
import asyncio
import re
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import OpenAIChatCompletionClient
from function import *
from kimi_web import *
# import nest_asyncio


# nest_asyncio.apply()
#
# gpt4_client = OpenAIChatCompletionClient(
#     model="gpt-3.5-turbo",
#     api_key=""
# )

# model="gpt-3.5-turbo",
# gpt-4o



termination = TextMentionTermination("结束")


def initialize_agent_team():
    gpt3_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key = ""
    )

    data_agent = AssistantAgent(
        name="data_agent",
        model_client=gpt3_client,
        tools=[fetch_stock_and_industry_data]
    )

    annual_report_agent = AssistantAgent(
        name="annual_report_agent",
        model_client=gpt3_client,
        tools=[extract_annual_report_content]
    )

    k_line_agent = AssistantAgent(
        name="k_line_analysis_agent",
        model_client=gpt3_client,
        tools=[k_line_read]
    )

    A_share_analyse_agent = AssistantAgent(
        name="A_share_analyse_agent",
        model_client=gpt3_client,
        tools=[analyse_A_markt]
    )

    business_model_agent = AssistantAgent(
        name="business_model_agent",
        model_client=gpt3_client,
        tools=[extract_business_and_development_info]
    )

    termination = TextMentionTermination("TERMINATE")

    return RoundRobinGroupChat(
        [business_model_agent,A_share_analyse_agent,
         annual_report_agent,
         data_agent,
         k_line_agent],
        termination_condition=termination
    )


async def run_task(task: str):

    agent_team = initialize_agent_team()

    stock_name = extract_stock_name(task)

    web_search = kimi_web_search(task)

    # print(f"Starting new task: {task}")

    task = task + '请解答上方用户提出的问题；你将使用以下5个agents进行该用户问题的解答；将使用的agent分别是如下' \
                  'business_model_agent；' \
                  'A_share_analyse_agent；' \
                  'k_line_agent；' \
                  'annual_report_agent；' \
                  'data_agent。' \
                  '请使用上方的agents.在最后完成该用户提出的问题。谢谢！'

    print(f"Starting new task: {task}")

    if not stock_name:
        return "Error: 未找到股票名称", "", "", None, "无法提取股票名称"

    k_line_image_path, k_line_prompt = await k_line_read(stock_name)



    full_result = ""
    A_share_analyse = ""
    data_report = ""
    annual_report = ""
    k_line_prompt = ""
    business_model_content = ''

    print(f"Stock Name: {stock_name}")
    print(f"A_share_analyse: {A_share_analyse}")
    print(f"K_line_prompt: {k_line_prompt}")
    print(f"Data Report: {data_report}")
    print(f"Annual Report: {annual_report}")
    print(f"business_model_content: {business_model_content}")

    try:

        stream = agent_team.run_stream(task=task)

        async for message in stream:
            print("-------message------", message)

            agent_name = getattr(message, 'source', None)
            content = getattr(message, 'content', None)

            if agent_name and content:

                content = str(content)
                print("________agent_name________", agent_name)
                print("________content__________", content)


                if not re.search(r'FunctionExecutionResult|FunctionCall', content):
                    print("-------进入分析处理----")

                    if agent_name == "A_share_analyse_agent":
                        A_share_analyse += content + "\n"
                        # print('A_share_analyse',A_share_analyse)
                    elif agent_name == "business_model_agent":
                        business_model_content += content + "\n"
                        # print("business_model_content",business_model_content)
                    elif agent_name == "k_line_analysis_agent":
                        k_line_prompt += content + "\n"
                        # print("k_line_prompt",k_line_prompt)
                    elif agent_name == "annual_report_agent":
                        annual_report += content + "\n"
                        # print('annual_report',annual_report)
                    elif agent_name == "data_agent":
                        data_report += content + "\n"
                        # print('data_report',data_report)



            else:
                print("Message does not contain expected attributes. Full message:", message)

        decision_response = decision_tool( stock_name, A_share_analyse,
                                           k_line_prompt, data_report,annual_report,business_model_content,web_search)

        full_result += decision_response


        final_decision_response = final_decision(stock_name, A_share_analyse,
                                                 k_line_prompt, data_report, annual_report, business_model_content,full_result,web_search)
        print("final_decision_response",final_decision_response)
        print("*"*50)


        decision_message, target_price = extract_decision_and_target_price(final_decision_response)
        print("decision_message",decision_message)
        print("*" * 50)
        print("target_price",target_price)
        print("*" * 50)


        decision_message = f"{decision_message}\n{target_price}"


        if os.path.exists(k_line_image_path):
            return full_result, A_share_analyse, data_report, \
                   annual_report, k_line_image_path, k_line_prompt, decision_message, business_model_content, web_search
        else:
            print("K线图文件未找到")
            return full_result, A_share_analyse, data_report, \
                   annual_report, None, k_line_prompt, decision_message, business_model_content, web_search

    except Exception as e:
        print(f"Error in task processing: {str(e)}")
        return f"Error: {str(e)}", "", "", None, "", ""



#
async def gradio_ui(task: str):

    result, A_share_analyse, financial_data, annual_report, \
    k_line_image_path, k_line_prompt, decision_message, business_model_content, web_search = await run_task(task)

    return result, A_share_analyse, financial_data, annual_report, k_line_prompt, \
           k_line_image_path, business_model_content, decision_message, web_search




def create_gradio_interface():
    with gr.Blocks() as demo:

        demo.css = """
            /* 调整文本框的字体和颜色 */
            #task-input { font-size: 15px; color: blue; font-weight: bold; }
            #output { font-size: 15px; color: green; font-weight: bold; }
            #A_share_analyse-output { font-size: 15px; color: SkyBlue; font-weight: bold; }
            #financial-data-output { font-size: 15px; color: orange; font-weight: bold; }
            #annual-report-output { font-size: 15px; color: purple; font-weight: bold; }
            #k-line-text-output { font-size: 15px; color: IndianRed; font-weight: bold; }
            #k_line_image_output { font-size: 15px; color: blue; font-weight: bold; }

            /* 设置“买卖决策和目标价格”框的位置和样式 */
            #decision-column {
                position: sticky;
                top: 20px;
            }
            #decision-output {
                font-size: 24px !important; /* 放大字体 */
                color: darkblue !important;
                font-weight: bold !important; /* 加粗字体 */
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #ccc;
            }

            /* 设置商业模式内容框 */
            #business-model-output {
                font-size: 18px !important;
                color: #007bff;
                font-weight: bold;
                padding: 10px;
                background-color: #f0f8ff;
                border-radius: 8px;
                border: 1px solid #ccc;
            }

            /* 设置web_search内容框 */
            #web_search {
                font-size: 18px !important;
                color: #007bff;
                font-weight: bold;
                padding: 10px;
                background-color: #f0f8ff;
                border-radius: 8px;
                border: 1px solid #ccc;
            }
        """

        gr.Markdown(
            "<h1 style='color: black; font-weight: bold; text-align: center; font-size: 48px;'>Auto -- Stock -- Agent</h1>",
            elem_id="header",
        )

        with gr.Row():
            with gr.Column(scale=3):
                # 用户输入框
                gr.Markdown(
                    "<h2 style='color: darkblue; font-weight: bold;'>请输入您的问题</h2>",
                    elem_id="task-input-header",
                )
                task_input = gr.Textbox(
                    label="",
                    placeholder="例如：请帮我分析一下股票贵州茅台，是否值得投资？",
                    elem_id="task-input",
                )

                # 最终决策结果
                gr.Markdown(
                    "<h2 style='color: IndianRed; font-weight: bold;'>最终决策结果</h2>",
                    elem_id="output-header",
                )
                output = gr.Textbox(
                    interactive=False,
                    elem_id="output",
                )

                # 大盘行情分析
                gr.Markdown(
                    "<h2 style='color: purple ; font-weight: bold;'>大盘行情分析</h2>",
                    elem_id="A_share_analyse-header",
                )
                A_share_analyse_output = gr.Textbox(
                    interactive=False,
                    elem_id="A_share_analyse-output",
                )

                # 股票行情分析
                gr.Markdown(
                    "<h2 style='color: orange; font-weight: bold;'>股票行情分析</h2>",
                    elem_id="financial-data-header",
                )
                financial_data_output = gr.Textbox(
                    interactive=False,
                    elem_id="financial-data-output",
                )

                # 年报基本面分析
                gr.Markdown(
                    "<h2 style='color: green; font-weight: bold;'>年报基本面分析</h2>",
                    elem_id="annual-report-header",
                )
                annual_report_output = gr.Textbox(
                    interactive=False,
                    elem_id="annual-report-output",
                )

                # 技术面透析
                gr.Markdown(
                    "<h2 style='color: red; font-weight: bold;'>技术面透析</h2>",
                    elem_id="k-line-header",
                )
                k_line_text_output = gr.Textbox(
                    interactive=False,
                    elem_id="k-line-text-output",
                )

                # 日K线图展示
                gr.Markdown(
                    "<h2 style='color: blue; font-weight: bold;'>日K线图展示</h2>",
                    elem_id="k-line-image-header",
                )
                k_line_image_output = gr.Image(
                    interactive=False,
                    elem_id="k_line_image_output",
                )


            with gr.Column(scale=1, elem_id="decision-column"):
                # 买卖决策和目标价格
                gr.Markdown(
                    "<h2 style='color: Red; font-weight: bold;'>买卖决策 & 目标价格</h2>",
                    elem_id="decision-header",
                )
                decision_output = gr.Textbox(
                    interactive=False,
                    elem_id="decision-output",
                    lines=2,  # 设置高度
                )

                # 商业模式分析内容展示
                gr.Markdown(
                    "<h2 style='color: #007bff; font-weight: bold;'>商业模式分析</h2>",
                    elem_id="business-model-header",
                )
                business_model_output = gr.Textbox(
                    interactive=False,
                    elem_id="business-model-output",
                    lines=5,  # 设置高度
                )

                # web_search内容展示
                gr.Markdown(
                    "<h2 style='color: #007bff; font-weight: bold;'>网络信息汇总</h2>",
                    elem_id="web_search-header",
                )
                web_search_output = gr.Markdown(
                    "",  # 初始为空
                    elem_id="web_search",

                )

        task_input.submit(
            gradio_ui,
            task_input,
            [
                output,
                A_share_analyse_output,
                financial_data_output,
                annual_report_output,
                k_line_text_output,
                k_line_image_output,
                business_model_output,
                decision_output,
                web_search_output
            ],
        )

        return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()


