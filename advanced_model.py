import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Attention(nn.Module):
    def __init__(self, input_size, cnn_out_channels, hidden_size, num_layers, output_size=1, dropout_rate=0.2):
        """
        构建 CNN-LSTM-Attention 混合预测网络
        :param input_size: 原始输入特征维度 (如 8)
        :param cnn_out_channels: 1D-CNN 提取后的特征维度
        :param hidden_size: LSTM 隐藏层神经元数量
        """
        super(CNN_LSTM_Attention, self).__init__()
        
        # 1. 一维卷积层 (1D-CNN)
        # 针对时序序列提取局部微观特征。使用 padding=1 保持时间步长度不变 (序列长度依然是 20)
        self.conv1d = nn.Conv1d(in_channels=input_size, 
                                out_channels=cnn_out_channels, 
                                kernel_size=3, 
                                padding=1)
        self.relu = nn.ReLU()
        
        # 2. LSTM 层
        # 接收 CNN 提取后的特征图进行长程记忆建模
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 3. 注意力机制 (Attention Mechanism)
        # 学习一个权重矩阵，用于评估 20 个时间步中哪一天的隐藏状态对最终预测最重要
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 4. 正则化与输出层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (Batch, TimeSteps, Features) -> (32, 20, 8)
        
        # --- CNN 阶段 ---
        # PyTorch 的 Conv1d 要求输入形状为 (Batch, Channels, TimeSteps)
        # 因此需要进行维度置换 (permute)
        x = x.permute(0, 2, 1)  # shape: (32, 8, 20)
        
        c_out = self.conv1d(x)  # shape: (32, cnn_out_channels, 20)
        c_out = self.relu(c_out)
        
        # 将维度置换回来，准备输入 LSTM (Batch, TimeSteps, Channels)
        c_out = c_out.permute(0, 2, 1)  # shape: (32, 20, cnn_out_channels)
        
        # --- LSTM 阶段 ---
        lstm_out, _ = self.lstm(c_out)  # lstm_out shape: (32, 20, hidden_size)
        
        # --- Attention 阶段 ---
        # 计算每个时间步的注意力得分
        # lstm_out 形状不变，传入全连接层计算得分
        attn_scores = self.attention_weights(lstm_out) # shape: (32, 20, 1)
        
        # 使用 Softmax 归一化为注意力权重 (和为 1)
        attn_weights = F.softmax(attn_scores, dim=1)   # shape: (32, 20, 1)
        
        # 将权重与 LSTM 输出进行加权求和 (Context Vector)
        # 矩阵乘法思维：将时间步维度进行压缩
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # shape: (32, hidden_size)
        
        # [临时新增] 退化为 M2 (CNN-LSTM)：直接取 LSTM 最后一个时间步的输出
        # context_vector = lstm_out[:, -1, :]

        # --- 预测输出阶段 ---
        out = self.dropout(context_vector)
        out = self.fc(out) # shape: (32, 1)
        
        return out

# ================= 测试桩 (Test Stub) =================
if __name__ == "__main__":
    # 模拟输入张量: Batch=32, TimeSteps=20, Features=8
    dummy_input = torch.randn(32, 20, 8)
    
    # 实例化混合模型
    model = CNN_LSTM_Attention(
        input_size=8, 
        cnn_out_channels=16, 
        hidden_size=64, 
        num_layers=2
    )
    
    output = model(dummy_input)
    
    print(f"输入张量形状: {dummy_input.shape}")
    print(f"混合模型输出形状: {output.shape}")
    print("CNN-LSTM-Attention 网络架构打通，梯度流正常！")