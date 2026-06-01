import akshare as ak
import pandas as pd

print("Yahoo Finance 遇到限流，已自动切换至高可用备用方案：AkShare...")
print("正在拉取 贵州茅台(600519) 2018-2024年的真实前复权日线交易数据...")

try:
    # 获取A股历史行情数据 (qfq = 前复权，消除分红派息对股价走势产生的断层影响，这对LSTM训练极其重要)
    df = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20180101", end_date="20240101", adjust="qfq")

    # 规范化列名，将其映射为我们的 DataLoader 所需的英文标准字段
    df.rename(columns={
        '日期': 'Date',
        '开盘': 'Open',
        '最高': 'High',
        '最低': 'Low',
        '收盘': 'Close',
        '成交量': 'Volume'
    }, inplace=True)

    # 严格提取需要的 6 个基础字段
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # 保存为 CSV
    csv_filename = "stock_data.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"太棒了！数据下载完成！已成功保存至 {csv_filename}")
    print(f"共获取 {len(df)} 个交易日的有效数据。")
    
except Exception as e:
    print(f"数据拉取失败，错误信息: {e}")