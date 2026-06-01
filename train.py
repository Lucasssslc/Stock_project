import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math

from data_loader import StockDataLoader
from advanced_model import CNN_LSTM_Attention

def train_and_evaluate():
    # ================= 1. 数据准备 =================
    print("正在加载并预处理一阶差分数据...")
    loader = StockDataLoader(file_path="stock_data.csv", sequence_length=20, train_split=0.8)
    # 【核心重构1】：接收多出来的 test_base_close
    X_train, y_train, X_test, y_test, test_base_close = loader.process_data()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # ================= 2. 挂载全新混合模型 =================
    print("正在初始化 CNN-LSTM-Attention 混合引擎...")
    model = CNN_LSTM_Attention(
        input_size=8, 
        cnn_out_channels=16, 
        hidden_size=64, 
        num_layers=2,
        dropout_rate=0.2
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    epochs = 150  # 增加轮数，让模型适应更难的差分预测
    
    # ================= 3. 核心训练循环 =================
    print(f"开始训练，共 {epochs} 轮 (预测目标为差分)...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), 'cnn_lstm_attn_model.pth')
        print("====== 虾哥提示：模型的大脑（权重）已成功保存到本地！ ======")
            
    # ================= 4. 模型评估 (逆向重构) =================
    print("训练完成，开始进行差分逆向重构与评估...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        
        pred_np = predictions.numpy()
        y_test_np = y_test_tensor.numpy()
        
        # 此时反归一化出来的，是真实的“涨跌金额”（比如 +15元 或 -8元）
        pred_diff_real = loader.scaler_y.inverse_transform(pred_np).flatten()
        y_test_diff_real = loader.scaler_y.inverse_transform(y_test_np).flatten()
        
        # 【核心重构2】：公式 P_t = P_{t-1} + Delta_P_t 
        # 将预测的涨跌幅，加到 t-1 时刻的真实基准价格上，还原出绝对股价！
        pred_real_price = test_base_close + pred_diff_real
        y_test_real_price = test_base_close + y_test_diff_real
        
        # 计算 RMSE 和 MAE (基于真实股价)
        mse = np.mean((pred_real_price - y_test_real_price) ** 2)
        rmse = math.sqrt(mse)
        mae = np.mean(np.abs(pred_real_price - y_test_real_price))
        
        print("\n" + "="*40)
        print("【差分重构版 CNN-LSTM-Attention】评估报告:")
        print(f"均方根误差 (RMSE): {rmse:.2f} 元")
        print(f"平均绝对误差 (MAE): {mae:.2f} 元")
        print("="*40)

        # ================= 5. 可视化对比 =================
        print("正在生成高保真预测对比图表...")
        import matplotlib.pyplot as plt
        
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(14, 7), dpi=300)
        
        display_days = 100
        y_test_plot = y_test_real_price[-display_days:]
        pred_plot = pred_real_price[-display_days:]
        
        plt.plot(y_test_plot, color='#1f77b4', linewidth=2, label='真实收盘价 (True Price)')
        plt.plot(pred_plot, color='#d62728', linewidth=2, linestyle='-', alpha=0.8, label='差分重构预测价')
        
        plt.title('基于差分重构的 CNN-LSTM-Attention 股价预测', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('时间步 (交易日)', fontsize=12)
        plt.ylabel('股票价格 (元)', fontsize=12)
        
        textstr = f'突破平滑陷阱后:\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
        props = dict(boxstyle='round', facecolor='#ffe6cc', alpha=0.8)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        plt.savefig('diff_cnn_lstm_attn_result.png')
        print("图表已保存为 diff_cnn_lstm_attn_result.png")
        plt.show()

if __name__ == "__main__":
    train_and_evaluate()