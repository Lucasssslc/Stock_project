import baostock as bs
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import time
import shutil
# 新增导入语句
from model import StockLSTM
from model import StockRNN
from data_loader import StockDataLoader
from advanced_model import CNN_LSTM_Attention

# ================= 配置核心实验股票池 =================
# 字典格式：{"股票代码": "股票名称"}
STOCKS_TO_TEST = {
    "600519": "贵州茅台_消费",
    "002594": "比亚迪_新能源",
    "002456": "立讯精密_科技",
    "000001": "平安银行_金融"
}

def fetch_stock_data(symbol, name):
    """
    【工业级重构】基于 Baostock 的数据拉取与本地缓存容灾机制
    对应论文 4.3.2 节：临时缓存结构设计
    """
    print(f"\n>>> 开始处理: {name} ({symbol})")
    
    # === 1. 容灾机制：检查本地缓存池 ===
    cache_file = f"cache_{symbol}.csv"
    if os.path.exists(cache_file):
        print(f"  -> [命中缓存] 发现本地已有 {name} 的高频缓存数据，直接读取，避开网络 I/O！")
        shutil.copyfile(cache_file, "temp_batch_data.csv")
        return True

    # === 2. Baostock 网络拉取 ===
    print(f"  -> [网络请求] 正在通过 Baostock 直连拉取 {name} 数据...")
    lg = bs.login() # 登录系统
    if lg.error_code != '0':
        print(f"  -> [报错] Baostock 登录失败: {lg.error_msg}")
        return False

    # Baostock 的股票代码需要带上前缀 (sh/sz)
    prefix = 'sh.' if symbol.startswith('6') else 'sz.'
    bs_code = prefix + symbol

    # 拉取日 K 线数据 (2代表前复权)
    rs = bs.query_history_k_data_plus(bs_code,
        "date,open,high,low,close,volume",
        start_date='2018-01-01', end_date='2024-01-01',
        frequency="d", adjustflag="2")

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    bs.logout() # 登出系统

    if not data_list:
        print(f"  -> [报错] 未能获取到 {name} 的有效数据。")
        return False

    # === 3. 数据清洗与格式对齐 ===
    df = pd.DataFrame(data_list, columns=rs.fields)
    df.rename(columns={'date':'Date', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
    
    # 关键：Baostock 返回的是字符串，必须强制转为浮点数张量，否则模型会报错
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna() # 剔除异常空值

    # === 4. 写入本地缓存池 ===
    df.to_csv(cache_file, index=False)
    df.to_csv("temp_batch_data.csv", index=False)
    print(f"  -> [拉取成功] {name} 数据已持久化至本地缓存池！")
    
    return True

def train_and_eval_single_stock(stock_name):
    """独立的训练与评估流水线"""
    loader = StockDataLoader(file_path="temp_batch_data.csv", sequence_length=20, train_split=0.8)
    X_train, y_train, X_test, y_test, test_base_close = loader.process_data()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=False)
    
    # 1. 注释掉原有的终极模型
    #model = CNN_LSTM_Attention(input_size=8, cnn_out_channels=16, hidden_size=64, num_layers=2, dropout_rate=0.2)
    # 2. 实例化纯 LSTM 模型作为基线 (M1)
    #model = StockLSTM(input_size=8, hidden_size=64, num_layers=2, dropout_rate=0.2)

    model = StockRNN(input_size=8, hidden_size=64, num_layers=2, dropout_rate=0.2)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    epochs = 200
    print(f"[{stock_name}] 模型训练中 (共{epochs}轮)...")
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    # 评估与逆向重构
    model.eval()
    with torch.no_grad():
        pred_np = model(X_test_tensor).numpy()
        y_test_np = y_test_tensor.numpy()
        
        pred_diff_real = loader.scaler_y.inverse_transform(pred_np).flatten()
        y_test_diff_real = loader.scaler_y.inverse_transform(y_test_np).flatten()
        
        pred_real_price = test_base_close + pred_diff_real
        y_test_real_price = test_base_close + y_test_diff_real
        
        rmse = math.sqrt(np.mean((pred_real_price - y_test_real_price) ** 2))
        mae = np.mean(np.abs(pred_real_price - y_test_real_price))
        
        print(f"[{stock_name}] 评估完毕 -> RMSE: {rmse:.2f} 元, MAE: {mae:.2f} 元")
        
# 计算朴素预测 (Naive Forecast) 基线误差
        naive_pred = y_test_real_price[:-1]  # 昨天的数据
        naive_actual = y_test_real_price[1:] # 今天的数据
        naive_rmse = math.sqrt(np.mean((naive_pred - naive_actual) ** 2))
        naive_mae = np.mean(np.abs(naive_pred - naive_actual))
        print(f"[{stock_name}] 朴素预测基线 -> RMSE: {naive_rmse:.2f} 元, MAE: {naive_mae:.2f} 元")

        # 自动生成图表并【静默保存】
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 6), dpi=200)
        
        display_days = 100
        plt.plot(y_test_real_price[-display_days:], color='#1f77b4', linewidth=2, label='真实收盘价')
        plt.plot(pred_real_price[-display_days:], color='#d62728', linewidth=2, linestyle='-', alpha=0.8, label='模型预测价')
        
        plt.title(f'{stock_name} - CNN-LSTM-Attention 股价预测', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('时间步 (交易日)')
        plt.ylabel('股票价格 (元)')
        plt.legend(loc='upper right')
        
        textstr = f'RMSE = {rmse:.2f}\nMAE = {mae:.2f}'
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#ffe6cc', alpha=0.8))
        
        # 关键：以股票名字命名文件，并关闭弹窗（避免阻塞循环）
        filename = f"Result_{stock_name}.png"
        plt.savefig(filename)
        plt.close() # 画完立刻关闭内存里的画布，进入下一轮
        print(f"[{stock_name}] 图表已保存为 {filename}\n")
        
        return rmse, mae

if __name__ == "__main__":
    print("====== 启动多标的自动化验证流水线 ======")
    results_log = {}
    
    for symbol, name in STOCKS_TO_TEST.items():
        success = fetch_stock_data(symbol, name)
        if success:
            rmse, mae = train_and_eval_single_stock(name)
            results_log[name] = {"RMSE": rmse, "MAE": mae}
            
    print("====== 所有实验运行完毕！评估汇总 ======")
    for name, metrics in results_log.items():
        print(f"{name}: RMSE = {metrics['RMSE']:.2f}, MAE = {metrics['MAE']:.2f}")
    
    # 清理临时文件
    if os.path.exists("temp_batch_data.csv"):
        os.remove("temp_batch_data.csv")