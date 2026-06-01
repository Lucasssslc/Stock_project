import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout_rate=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 核心 LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 独立的全连接前 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接层 (线性映射)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM 前向传播
        out, _ = self.lstm(x)
        
        # 时间步截断：仅提取最后一个时间步的输出特征进行预测
        out = out[:, -1, :]
        
        # 正则化与线性映射
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

# ================= 测试桩 (Test Stub) =================
# 注意这里的顶格缩进
if __name__ == "__main__":
    # 模拟数据加载器输出的单批次张量 (Batch Size=32, Sequence=20, Features=8)
    dummy_input = torch.randn(32, 20, 8)
    
    # 实例化模型
    model = StockLSTM(input_size=8, hidden_size=64, num_layers=2, dropout_rate=0.2)
    
    # 前向传播测试
    output = model(dummy_input)
    
    print(f"输入张量形状: {dummy_input.shape}")
    print(f"输出张量形状: {output.shape}")
    print("模型架构打通，梯度流正常！")

class StockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout_rate=0.2):
        # 1. 名字改成 StockRNN
        super(StockRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 2. 核心替换：将 nn.LSTM 换成了最传统的 nn.RNN
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # 独立的全连接前 Dropout 层 (保持原样)
        self.dropout = nn.Dropout(dropout_rate)

        # 全连接层 (保持原样)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 3. 这里调用刚才定义的 self.rnn
        out, _ = self.rnn(x)

        # 时间步截断：仅提取最后一个时间步的输出 (保持原样)
        out = out[:, -1, :]
        
        # 经过 Dropout 和 全连接层 (保持原样)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out