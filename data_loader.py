import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class StockDataLoader:
    def __init__(self, file_path, sequence_length=20, train_split=0.8):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.train_split = train_split
        # 独立的缩放器，防止特征与目标量纲污染
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        # 目标缩放器现在针对的是“涨跌差额”
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        
    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _feature_engineering(self, df):
        df = df.sort_values('Date').reset_index(drop=True)
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI_14'] = self._compute_rsi(df['Close'], period=14)
        
        # 【核心重构】：计算一阶差分作为预测目标（消除非平稳性趋势）
        df['Diff_Close'] = df['Close'].diff()
        
        # 丢弃因滚动窗口和差分产生的 NaN 行
        df = df.dropna().reset_index(drop=True)
        return df

    def process_data(self):
        df = pd.read_csv(self.file_path)
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        df = self._feature_engineering(df)
        all_features = features + ['MA_5', 'MA_20', 'RSI_14']
        
        data_X = df[all_features].values
        # 目标变成了差分列！
        data_y = df['Diff_Close'].values.reshape(-1, 1) 
        
        # 提取用于后续还原绝对股价的基准序列 (即真实的 Close)
        close_prices = df['Close'].values
        
        split_idx = int(len(data_X) * self.train_split)
        
        X_train, X_test = data_X[:split_idx], data_X[split_idx:]
        y_train, y_test = data_y[:split_idx], data_y[split_idx:]
        
        # 严格隔离的归一化
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # 滑动窗口切片内部函数，支持基准价格对齐
        def create_sequences(data, labels, close_arr, start_idx):
            X, y, base_close = [], [], []
            for i in range(self.sequence_length, len(data)):
                X.append(data[i - self.sequence_length : i])
                y.append(labels[i])
                # 记录 t-1 时刻的真实收盘价，极其关键！用于最后还原 t 时刻的绝对价格
                base_close.append(close_arr[start_idx + i - 1]) 
            return np.array(X), np.array(y), np.array(base_close)

        X_train_seq, y_train_seq, _ = create_sequences(X_train_scaled, y_train_scaled, close_prices, 0)
        # 测试集额外返回 test_base_close
        X_test_seq, y_test_seq, test_base_close = create_sequences(X_test_scaled, y_test_scaled, close_prices, split_idx)
        
        return X_train_seq, y_train_seq, X_test_seq, y_test_seq, test_base_close

# ================= 测试桩 =================
if __name__ == "__main__":
    print("正在测试一阶差分数据加载器...")
    loader = StockDataLoader(file_path="stock_data.csv", sequence_length=20, train_split=0.8)
    X_train, y_train, X_test, y_test, test_base = loader.process_data()
    print(f"训练集张量 X: {X_train.shape}, 目标 y(差分): {y_train.shape}")
    print(f"测试集张量 X: {X_test.shape}, 目标 y(差分): {y_test.shape}, 基准价格: {test_base.shape}")
    print("底层数据流重构成功！目标已转化为一阶差分！")