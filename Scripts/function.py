from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import tushare as ts
import matplotlib.dates as mdates
import pandas as pd
import asyncio
import base64
import time
import os
import akshare as ak
import traceback
import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import Console, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import OpenAIChatCompletionClient
from pylab import mpl
import pdfplumber
from datetime import datetime
from datetime import timedelta
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']



ts.set_token('')
pro = ts.pro_api()

api_key = ""

def extract_stock_name(question: str) -> str:

    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    matching_stocks = data[data['name'].apply(lambda name: name in question)]


    if not matching_stocks.empty:
        stock_name = matching_stocks['name'].iloc[0]
        print("匹配的股票名称:", stock_name)
        return stock_name
    else:
        print("没有找到匹配的股票名称")
        return ""



def analyse_A_markt(stock_name: str) -> str:
    """
    分析上证指数（000001.SH）的大盘走势，包括涨跌幅、估值与收益率差值，以及均线分析。
    返回一个包含大盘分析和趋势判断的prompt。
    """
    import traceback

    try:
        print(f"analyse_A_markt called with stock_name: {stock_name}")
        print("-------------------进入预判上证指数-------------------")
        code = '000001.SH'
        start_date = '2024-01-01'


        df1 = pro.index_dailybasic(ts_code=code, start_date=start_date, fields='ts_code,trade_date,pe')
        df2 = pro.index_daily(ts_code=code)


        df1 = df1[['ts_code', 'trade_date', 'pe']]
        df2 = df2[['ts_code', 'trade_date', 'close']]

        df1['trade_date'] = pd.to_datetime(df1['trade_date'], format='%Y%m%d')
        df2['trade_date'] = pd.to_datetime(df2['trade_date'], format='%Y%m%d')

        df3 = pd.merge(df1, df2, on=['ts_code', 'trade_date'])
        df3 = df3.sort_values(by=['trade_date'], ascending=True)


        dd1 = pro.index_dailybasic(ts_code='000001.SH')
        dd1['trade_date'] = pd.to_datetime(dd1['trade_date'], format='%Y%m%d')
        dd1 = dd1.sort_values(by=['trade_date']).reset_index(drop=True)
        dd1['1/PE'] = 1 / dd1['pe_ttm'].astype(float)


        dd2 = ak.bond_zh_us_rate()
        dd2 = dd2[['日期', '中国国债收益率10年', '美国国债收益率10年']]
        dd2 = dd2.sort_values(by=['日期']).reset_index(drop=True)
        dd2['trade_date'] = pd.to_datetime(dd2['日期'], format='%Y-%m-%d')
        dd2 = dd2.ffill()  # 替换 fillna(method='ffill')


        dd = pd.concat([dd1[['1/PE', 'trade_date']].set_index('trade_date'),
                        dd2[['中国国债收益率10年', '美国国债收益率10年', 'trade_date']].set_index('trade_date')],
                       axis=1)
        dd['上证估值与中国国债收益率差值'] = dd['1/PE'] - dd['中国国债收益率10年'].astype(float) / 100
        dd['上证估值与美国国债收益率差值'] = dd['1/PE'] - dd['美国国债收益率10年'].astype(float) / 100
        dd = dd[['1/PE', '上证估值与中国国债收益率差值', '上证估值与美国国债收益率差值']].astype(float)
        dd = dd.bfill()  # 替换 fillna(method='bfill')
        dd.index = pd.to_datetime(dd.index)
        dd = dd[dd.index >= pd.Timestamp('2013-01-01')]


        df4 = pd.merge(df3, dd.reset_index(), on=['trade_date'])
        # print("--------------df4------------", df4)


        df4 = df4.sort_values(by=['trade_date'])
        df4['10_day_return'] = df4['close'].pct_change(10).fillna(0)
        df4['30_day_return'] = df4['close'].pct_change(30).fillna(0)
        df4['60_day_return'] = df4['close'].pct_change(60).fillna(0)


        df4['ma_5'] = df4['close'].rolling(window=5).mean()
        df4['ma_10'] = df4['close'].rolling(window=10).mean()
        df4['ma_20'] = df4['close'].rolling(window=20).mean()


        latest = df4.iloc[-1]
        prev = df4.iloc[-2]

        if latest['ma_5'] > latest['ma_10']:
            cross_status = "金叉"
        elif latest['ma_5'] < latest['ma_10']:
            cross_status = "死叉"
        else:
            cross_status = "无明显趋势"


        avg_difference = dd['上证估值与中国国债收益率差值'].mean()
        recent_difference = latest['上证估值与中国国债收益率差值']
        valuation_status = (
            "偏高" if recent_difference > avg_difference else "偏低"
        )


        prompt = (
            f"你是一个优秀的金融分析师。以下是上证指数的近期数据情况:\n"
            f"上证指数最近10天涨幅：{latest['10_day_return'] * 100:.2f}%\n"
            f"最近30天涨幅：{latest['30_day_return'] * 100:.2f}%\n"
            f"最近60天涨幅：{latest['60_day_return'] * 100:.2f}%\n"
            f"5日均线：{latest['ma_5']:.2f}，10日均线：{latest['ma_10']:.2f}，20日均线：{latest['ma_20']:.2f}\n"
            f"当前均线状态：{cross_status}\n"
            f"当前的上证估值与中国国债收益率差值为：{recent_difference:.2f}，"
            f"历史平均值为：{avg_difference:.2f}，属于{valuation_status}水平。\n"
            f"基于以上数据，请分析当前上证指数是否存在风险还是机会。"
            f"切记！请确保不要在你的输出内容中出现： TERMINATE!！可以吗？谢谢！！"
        )

        print("大盘的prompt",prompt)



        return prompt

    except Exception as e:

        print("Error occurred in analyse_A_markt:")
        traceback.print_exc()
        return f"Error: {str(e)}"





def fetch_stock_and_industry_data(stock_name: str) -> str:
    """
    分析指定股票和其行业数据。
    :param stock_name: 股票名称
    :return: 分析结论
    """
    print("-------------------进入股票数据获取-------------------")
    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    ts_code = data.loc[data['name'] == stock_name, 'ts_code'].values[0]

    print('股票数据代码',ts_code)

    today = datetime.strftime(datetime.now(), '%Y%m%d')


    try:

        trade_data = ts.pro_bar(ts_code=ts_code, start_date='20240601', end_date=today, freq='D')
        # print("-----trade_data-----",trade_data)
        if trade_data is None or trade_data.empty:
            return f"No trading data available for {ts_code}."


        daily_basic_data = pro.daily_basic(ts_code=ts_code, trade_date=today, fields='pe,pb')
        if daily_basic_data is None or daily_basic_data.empty:
            return f"No daily basic data available for {ts_code}."


        financial_data = pro.query('fina_indicator', ts_code=ts_code, start_date='20240601', end_date=today)
        if financial_data is None or financial_data.empty:
            return f"No financial data available for {ts_code}."


        trade_data = trade_data.sort_values(by='trade_date')
        trade_data['cum_return_60d'] = trade_data['close'].pct_change(60).fillna(0)
        trade_data['cum_return_30d'] = trade_data['close'].pct_change(30).fillna(0)
        trade_data['cum_return_10d'] = trade_data['close'].pct_change(10).fillna(0)


        industry_members = pro.index_member_all(l1_code='')
        industry_row = industry_members[industry_members['ts_code'] == ts_code]
        if industry_row.empty:
            return f"No industry information found for stock {ts_code}."

        l1_code = industry_row.iloc[0]['l1_code']
        l1_name = industry_row.iloc[0]['l1_name']
        print("-----l1_code-----",l1_code)
        print("-----l1_name-----",l1_name)


        # industry_data = pro.sw_daily(fields='')
        # industry_data = industry_data[industry_data['ts_code'] == l1_code].sort_values(by='trade_date')
        industry_data = pro.sw_daily(ts_code = l1_code,fields='').sort_values(by='trade_date')
        # print("industry_data",industry_data )
        if industry_data.empty:
            return f"No industry data available for {l1_name} ({l1_code})."


        industry_data['cum_return_60d'] = industry_data['pct_change'].rolling(60).sum()
        industry_data['cum_return_30d'] = industry_data['pct_change'].rolling(30).sum()
        industry_data['cum_return_10d'] = industry_data['pct_change'].rolling(10).sum()


        stock_result = {
            "recent_60d_return": trade_data['cum_return_60d'].iloc[-1],
            "recent_30d_return": trade_data['cum_return_30d'].iloc[-1],
            "recent_10d_return": trade_data['cum_return_10d'].iloc[-1],
            "pe": daily_basic_data['pe'].iloc[0],
            "pb": daily_basic_data['pb'].iloc[0],
            "financial_data": financial_data.to_dict()
        }

        industry_result = {
            "recent_60d_return": industry_data['cum_return_60d'].iloc[-1],
            "recent_30d_return": industry_data['cum_return_30d'].iloc[-1],
            "recent_10d_return": industry_data['cum_return_10d'].iloc[-1],
            "pe": industry_data['pe'].mean(),
            "pb": industry_data['pb'].mean()
        }


        # report = (
        #     f"Analysis for stock {ts_code}:\n"
        #     f"60-Day Cumulative Return: {stock_result['recent_60d_return']*100:.2f}%\n"
        #     f"30-Day Cumulative Return: {stock_result['recent_30d_return']*100:.2f}%\n"
        #     f"10-Day Cumulative Return: {stock_result['recent_10d_return']*100:.2f}%\n"
        #     f"PE: {stock_result['pe']:.2f}\n"
        #     f"PB: {stock_result['pb']:.2f}\n"
        #     f"Financial Data: {stock_result['financial_data']}\n\n"
        #     f"Industry ({l1_name}) Analysis:\n"
        #     f"60-Day Cumulative Return: {industry_result['recent_60d_return']*100:.2f}%\n"
        #     f"30-Day Cumulative Return: {industry_result['recent_30d_return']*100:.2f}%\n"
        #     f"10-Day Cumulative Return: {industry_result['recent_10d_return']*100:.2f}%\n"
        #     f"Average PE: {industry_result['pe']:.2f}\n"
        #     f"Average PB: {industry_result['pb']:.2f}\n\n"
        #     f"请根据上方的数据信息，对这个股票进行详细的分析。谢谢您的帮助！"
        # )


        prompt = (
            f"你是一个资深的金融分析师，以下是关于股票{ts_code}和其所属行业的一些数据：\n\n"
            f"股票数据:\n{stock_result}\n\n"
            f"行业数据:\n{industry_result}\n\n"
            "请按照以下步骤进行分析，并给出详细的推理链：\n"
            "1. 首先，分析股票的当前财务状况（PE，PB，累计涨跌幅等），并与行业的平均水平进行对比。\n"
            "2. 接下来，分析行业的整体表现，以及它是否存在某些显著的趋势。\n"
            "3. 结合这些数据，给出你对该股票未来走势的预期。\n"
            # "4. 如果你认为股票的表现可能不如预期，提供改进建议。\n\n"
            "请在分析中清晰地展示你的思路和结论，确保每一步都经过充分推理。"
            f"切记！！！请确保不要在你输出内容中出现：TERMINATE! "
            f"可以吗？谢谢！"

        )
        # print("prompt",prompt)
        return prompt

    except Exception as e:
        return f"Error while processing data for {ts_code}: {str(e)}"


def extract_annual_report_content(stock_name: str) -> str:
    """
    提取年报中的指定内容，并进行总结。
    :param stock_name: 股票名称
    :return: 提取的年报内容摘要
    """
    # print("----------stock_name-----------", stock_name)
    print("-------------------进入年报数据获取-------------------")
    try:

        pdf_path = f"/Users/jacky/Desktop/年报/{stock_name}-年报.pdf"
        print("pdf_path", pdf_path)
        if not os.path.exists(pdf_path):
            return f"未找到{stock_name}的年报文件，请检查路径。"


        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()

        if stock_name.lower() not in full_text.lower():
            return "无法找到该股票的年报内容。"

        sections = ["经营情况讨论与分析", "报告期内公司所处行业情况", "报告期内公司从事的业务情况",
                    "报告期内核心竞争力分析", "公司关于公司未来发展的讨论与分析"]
        extracted_content = []
        for section in sections:
            start_idx = full_text.lower().find(section.lower())
            if start_idx != -1:
                end_idx = full_text.find("第三节", start_idx + 1)
                section_content = full_text[start_idx:end_idx].strip()
                extracted_content.append(f"{section}:\n{section_content}\n")


        summary = "\n".join(extracted_content)
        summary = summary[:6000]
        print("-------------获得的年报长度-------------", len(summary))


        prompt = (
            f"你是一个经验丰富的财务分析师，以下是关于股票{stock_name}的年报内容：\n\n"
            f"年报内容:\n{summary}\n\n"
            "请根据上述年报内容，进行详细分析，并给出你对该公司未来发展潜力的评估。"
            "请注意，在分析中考虑公司业务模式、行业竞争力、财务健康状况以及未来的增长潜力。"
            f"切记！！！请确保不要在你输出内容中出：TERMINATE!，谢谢！"
            # "请结合年报中的具体数字和趋势，给出对该股票的投资建议。"

        )

        # print("年报部分的prompt",prompt)

        return prompt
    except Exception as e:
        return f"Error while extracting content for {stock_name}: {str(e)}"





def calculate_macd(df):

    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def calculate_kdj(df):

    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()

    df['rsv'] = (df['close'] - low_min) / (high_max - low_min) * 100
    df['k'] = df['rsv'].ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    return df


def calculate_bollinger_bands(df):

    df['bollinger_middle'] = df['close'].rolling(window=20).mean()
    df['bollinger_upper'] = df['bollinger_middle'] + 2 * df['close'].rolling(window=20).std()
    df['bollinger_lower'] = df['bollinger_middle'] - 2 * df['close'].rolling(window=20).std()
    return df

def plot_k_line(stock_name: str, start_date: str, end_date: str):
    print("-------------------进入画图模式-----------------")


    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    ts_code = data.loc[data['name'] == stock_name, 'ts_code'].values[0]


    df = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date, adj='qfq')
    df = df.sort_values(by='trade_date')
    df['trade_date'] = pd.to_datetime(df['trade_date'])


    df = calculate_macd(df)
    df = calculate_kdj(df)
    df = calculate_bollinger_bands(df)

    fig = plt.figure(figsize=(14, 10), dpi=100)


    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(df['trade_date'], df['close'], label='Close Price', color='black')
    ax1.plot(df['trade_date'], df['close'].rolling(window=5).mean(), label='5-Day MA', color='orange')
    ax1.plot(df['trade_date'], df['close'].rolling(window=10).mean(), label='10-Day MA', color='green')
    ax1.plot(df['trade_date'], df['close'].rolling(window=20).mean(), label='20-Day MA', color='blue')
    ax1.plot(df['trade_date'], df['close'].rolling(window=60).mean(), label='60-Day MA', color='purple')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.set_title(f'K-Line Chart for {stock_name} ({ts_code})')
    ax1.grid(True)

    ax1_volume = ax1.twinx()
    ax1_volume.bar(df['trade_date'], df['vol'] / 1e6, color='lightblue', alpha=0.3, label='Volume')
    ax1_volume.set_ylabel('Volume (Million)')
    ax1_volume.legend(loc='upper right')


    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(df['trade_date'], df['macd'], label='MACD', color='blue')
    ax2.plot(df['trade_date'], df['macd_signal'], label='Signal Line', color='red')
    ax2.bar(df['trade_date'], df['macd_hist'], label='MACD Histogram', color='grey', alpha=0.5)
    ax2.set_ylabel('MACD')
    ax2.legend(loc='upper left')
    ax2.grid(True)


    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(df['trade_date'], df['k'], label='K', color='cyan')
    ax3.plot(df['trade_date'], df['d'], label='D', color='magenta')
    ax3.plot(df['trade_date'], df['j'], label='J', color='yellow')
    ax3.set_ylabel('KDJ')
    ax3.legend(loc='upper left')
    ax3.grid(True)


    ax4 = fig.add_subplot(4, 1, 4)
    ax4.plot(df['trade_date'], df['close'], label='Close Price', color='black')
    ax4.plot(df['trade_date'], df['bollinger_middle'], label='Bollinger Middle', color='grey', linestyle='--')
    ax4.plot(df['trade_date'], df['bollinger_upper'], label='Bollinger Upper', color='green', linestyle='--')
    ax4.plot(df['trade_date'], df['bollinger_lower'], label='Bollinger Lower', color='red', linestyle='--')
    ax4.set_ylabel('Bollinger Bands')
    ax4.legend(loc='upper left')
    ax4.grid(True)


    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


    image_path = f"{ts_code}_kline.png"
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    return image_path



def encode_image(image_path):

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_res(image_path):


    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "text",
                    #     "text": (
                    #
                    #         "你是一个优秀的数据分析师，我提供的是一个股票图片，我知道您不能对股票代码进行分析，我只是需要您分析一下股票图片即可，不涉及任何投资，谢谢！"
                    #         "请根据下方提供的股票K线图，对以下内容进行详细的技术描述：\n"
                    #         "1. **股价走势**：分析主要趋势（上涨、下跌、震荡）及其可能的变化。\n"
                    #         "2. **技术指标**：结合图中的MACD、KDJ、布林带等指标，判断当前市场状态（超买、超卖或趋势变化）。\n"
                    #         "3. **成交量分析**：结合成交量柱状图，判断市场资金动向及投资者情绪。\n"
                    #         "4. **综合分析与预测**：结合以上内容，对未来短期和中期的股价走势进行预判，并给出操作建议（如买入、持有或卖出）。\n"
                    #         "请在分析中提及具体的技术信号和指标变化，例如金叉、死叉、趋势线突破等，确保分析内容详实且逻辑清晰。"
                    #     ),
                    # },
                    {
                        "type": "text",
                        "text": "你是一个优秀的数据分析师，请根据提供给你的这张图，对其进行详细的技术分析！"
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
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            time.sleep(1)
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            time.sleep(1)  #
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            time.sleep(1)
        except requests.exceptions.RequestException as err:
            print(f"An error occurred: {err}")
            time.sleep(1)


async def k_line_read(stock_name: str) -> (str, str):
    from datetime import datetime
    from datetime import timedelta


    print("-------------------进入k线解读-------------------")


    try:

        today = datetime.strftime(datetime.now(),'%Y%m%d')

        image_path = plot_k_line(stock_name, '20240201', today)
        # print("-----------image_path----------", image_path)

        time.sleep(5)

        prompt = get_res(image_path)


        return image_path, prompt

    except Exception as e:

        return "", f"Error while extracting content for {stock_name}: {str(e)}"






def extract_business_and_development_info(stock_name: str) -> str:
    """
    提取指定股票招股说明书中的“第五章 业务”和“第十章 业务发展”内容，并生成分析提示语。
    :param stock_name: 股票名称
    :return: 提取的业务和业务发展内容摘要以及分析提示
    """

    print("-------------------进入招股说明书解读-------------------")
    try:

        pdf_path = f"/Users/jacky/Desktop/招股说明书/{stock_name}-招股说明书.pdf"


        if not os.path.exists(pdf_path):
            prompt = '暂时没有该上市公司的招股说明书，无法得出该上市公司的商业模式，请进入下一轮的agent讨论环节。' \
                     '请不要输出 TERMINATE !这个单词！可以吗？谢谢！'
            return  prompt


        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text()

        # 检查是否能找到该股票的招股说明书内容
        if stock_name.lower() not in full_text.lower():
            prompt = '暂时没有该上市公司的招股说明书，无法得出该上市公司的商业模式，请进入下一轮的agent讨论环节。跳过这股环节讨论。谢谢！请不要输出 TERMINATE !这个单词！可以吗？谢谢！'
            return prompt
        try:
            # 查找“第五章 业务”和“第十章 业务发展”章节
            sections = ["第五章 业务", "第十章 业务发展"]
            extracted_content = []

            for section in sections:
                # print("section", section)
                start_idx = full_text.lower().find(section.lower())
                if start_idx != -1:
                    # 查找下一个章节的开始，作为当前章节的结束
                    end_idx = full_text.find("第", start_idx + 1)
                    section_content = full_text[start_idx:end_idx].strip()
                    extracted_content.append(f"{section}:\n{section_content}\n")


            summary = "\n".join(extracted_content)
            # print('summary', summary)
            summary = summary[:8000]  # 限制输出长度


            prompt = (
                f"你是一个经验丰富的商业分析师，以下是关于企业{stock_name}的招股说明书内容：\n\n"
                f"招股说明书内容:\n{summary}\n\n"
                "请根据上述招股说明书内容，进行详细分析，并给出你对该公司商业模式及行业竞争力的评估。"
                "请在分析中考虑公司业务模式、行业地位、市场前景及竞争对手的情况。"
                "请注意，当下只是分析企业的商业模型和竞争对手，暂时不需要提供买卖建议！"
                f"切记！！！切记！请确保不要在你最后输出内容中出现 ： TERMINATE !这个单词！可以吗？谢谢！"


            )
            print("招股说明书prompt", prompt)

            return prompt

        except:
            prompt = '暂时没有该上市公司的招股说明书，无法得出该上市公司的商业模式，请进入下一轮的agent讨论环节。跳过这股环节讨论。谢谢！请不要输出 TERMINATE !这个单词！可以吗？谢谢！'
            return prompt

    except Exception as e:
        prompt = '暂时没有该上市公司的招股说明书，无法得出该上市公司的商业模式，请进入下一轮的agent讨论环节。'
        return f"Error while extracting content for {stock_name}: {str(e)}"





def decision_tool( stock_name: str,
                   A_share_analyse,
                   k_line_prompt: str,
                   data_report: str,
                   annual_report: str,
                   business_model_content: str,
                   web_search: str) -> str:
    print("-------------------进入决策阶段-------------------")


    prompt = (
        f"你是一个专业的金融分析师。根据下方分析的技术面走势分析、基本面数据分析和年报分析，"
        f"做出是否买入或卖出的{stock_name}决策，并解释推理过程：\n\n"
        f"当前整体的上证指数环境情况报告:\n{A_share_analyse}\n\n"
        f"技术面分析报告：\n{k_line_prompt}\n\n"
        f"基本面分析报告：\n{data_report}\n\n"
        f"商业模式：\n{business_model_content}\n\n"
        f"年报内容：\n{annual_report}\n\n"
        f"企业最近新闻动态：\n{web_search}\n\n"
        "推理过程：\n"
        "1. 首先从整体的上证指数走势情况报告分析，上证指数短期的存在风险还是机会，对该股票的后期影响。\n"
        "2. 首先从K线图走势中分析股票走势行情，预判该股票的后期走势，"
        "短期是否存在机会或者风险。\n"
        "3. 结该公司的新闻动态，分析公司当下的社会舆情和分析是否偏向有利的方面。\n"
        "4. 从财务报告中提取关键的财务指标，分析该股票的财务健康状况。\n"
        "5. 结合商业模式，分析公司在行业中的位置、未来发展规划等。\n"
        "6. 结合年报内容，分析公司当下的经营情况是否处于上升周期，未来的业绩是否可以支撑当下的股价。\n"
        "7. 最后综合以上分析，得出是否买入、卖出、持有或观望的决策。\n\n"
        "请根据这些信息给出明确的决策，并解释推理过程。"
    )

    # print("Final prompt length:", len(prompt))
    # print("prompt", prompt)


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }


    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 3000
    }


    while True:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            decision_text = response.json()['choices'][0]['message']['content']
            # print("decision_text",decision_text)
            return decision_text.strip()
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





def final_decision(stock_name: str, A_share_analyse:str, k_line_prompt: str,
                   data_report: str,
                   annual_report: str,
                   business_model_content: str,
                   full_result: str,
                   web_search: str) -> str:
    print("-------------------进入目标价格决策-------------------")


    prompt = (
        f"基于以下信息，判断{stock_name}是否值得买入，并给出目标价格："
        f"A股上证指数整体分析：{A_share_analyse};"
        f"企业最近的新闻动态分析：{web_search};"
        f"企业市场数据：{data_report};"
        f"企业商业模型分析：{business_model_content};"
        f"企业年报分析：{annual_report};"
        f"企业K线分析：{k_line_prompt};"
        f"综合分析：{full_result};"
        f"请给出短期买/不买的决策，并必须提供一个短期目标价格。"
        f"请您最后的回答，只需要提供：短期买卖决策：""，短期目标价格：""；即可，谢谢您"
        # f"请您最后的回答，只需要提供：短期（1个月内）买卖决策：""，短期（1个月内）目标价格：""；"
        # f"中期（半年后）买卖决策：""，中期（半年内）目标价格：""；"
        # f"长期（1年后）买卖决策：""，长期（1年后）目标价格：""；"
        # "请提供上方三种情况目标价格和买卖决策即可，谢谢您！"
    )

    # print("Final prompt length:", len(prompt))
    # print("prompt", prompt)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 3000
    }

    while True:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            decision_text = response.json()['choices'][0]['message']['content']
            print("final_decison", decision_text)
            return decision_text.strip()
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

#
def extract_decision_and_target_price(final_decision_response: str):

    cleaned_response = final_decision_response.strip()
    print("cleaned_response",cleaned_response)

    if not cleaned_response:
        return "Error: Empty response"

    response_parts = cleaned_response.split("\n")
    print("response_parts",response_parts)

    if len(response_parts) < 2:
        return "Error: Response format is not correct."


    decision_message =  response_parts[0].strip()
    target_price = response_parts[1].strip()

    return decision_message, target_price




# import re
#
# def extract_decision_and_target_price(final_decision_response: str):
#     """
#     从final_decision_response中提取短期、中期和长期的买卖决策及目标价格。
#     :param final_decision_response: 来自final_decision函数的完整决策文本
#     :return: 包含决策和目标价格的元组（短期决策，短期目标股价，中期决策，中期目标股价，长期决策，长期目标股价）
#     """
#     # 清理响应文本
#     cleaned_response = final_decision_response.strip()
#     print("cleaned_response:", cleaned_response)
#
#     if not cleaned_response:
#         return "Error: Empty response"
#
#     # 使用正则表达式提取各个时间段的决策和目标价格
#     short_term_pattern = r"短期（1个月内）买卖决策：\"(.*?)\"，短期（1个月内）目标价格：\"(.*?)\""
#     mid_term_pattern = r"中期（半年后）买卖决策：\"(.*?)\"，中期（半年内）目标价格：\"(.*?)\""
#     long_term_pattern = r"长期（1年后）买卖决策：\"(.*?)\"，长期（1年后）目标价格：\"(.*?)\""
#
#     # 提取短期、中期、长期的决策和目标股价
#     short_term_match = re.search(short_term_pattern, cleaned_response)
#     mid_term_match = re.search(mid_term_pattern, cleaned_response)
#     long_term_match = re.search(long_term_pattern, cleaned_response)
#
#     # 如果找到了相应的目标股价和决策，提取内容
#     short_term_decision, short_term_target_price = (short_term_match.group(1), short_term_match.group(2)) if short_term_match else ("未定义", "未定义")
#     mid_term_decision, mid_term_target_price = (mid_term_match.group(1), mid_term_match.group(2)) if mid_term_match else ("未定义", "未定义")
#     long_term_decision, long_term_target_price = (long_term_match.group(1), long_term_match.group(2)) if long_term_match else ("未定义", "未定义")
#
#     return (short_term_decision, short_term_target_price,
#             mid_term_decision, mid_term_target_price,
#             long_term_decision, long_term_target_price)
