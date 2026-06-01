from flask import Flask, render_template, jsonify, request
import torch
import numpy as np
import baostock as bs
import pandas as pd
import os
from datetime import datetime, timedelta

# 导入你的算法模块
from data_loader import StockDataLoader
from advanced_model import CNN_LSTM_Attention

app = Flask(__name__)

# ================= 1. 全局挂载 AI 模型 =================
print("正在将 CNN-LSTM-Attention 模型载入服务器内存...")
model = CNN_LSTM_Attention(input_size=8, cnn_out_channels=16, hidden_size=64, num_layers=2, dropout_rate=0.2)
# 加载你炼好的大脑权重
model.load_state_dict(torch.load('cnn_lstm_attn_model.pth', weights_only=True))
model.eval()
print("模型加载完毕，AI 引擎已就绪！")

@app.route('/')
def home():
    return render_template('index.html')

# ================= 2. 核心预测 API =================
@app.route('/api/predict', methods=['GET'])
def predict_api():
    try:
        stock_code = request.args.get('code', '600519')
        print(f"收到前端请求，准备使用 Baostock 预测股票：{stock_code}")

        # 智能拼装 Baostock 代码前缀
        if stock_code.startswith('6'):
            bs_code = f"sh.{stock_code}"
        elif stock_code.startswith('0') or stock_code.startswith('3'):
            bs_code = f"sz.{stock_code}"
        else:
            bs_code = f"sh.{stock_code}"

        # 动态获取真实日期 (今天，以及两年前)
        today = datetime.now().strftime('%Y-%m-%d')
        two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        # 登录并拉取数据
        bs.login()
        rs = bs.query_history_k_data_plus(bs_code,
            "date,open,high,low,close,volume",
            start_date=two_years_ago, end_date=today,
            frequency="d", adjustflag="2")
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()

        if not data_list:
            return jsonify({"status": "error", "message": f"Baostock 未找到该股票数据，请检查代码是否正确: {stock_code}"})

        # 数据格式化与类型转换
        df = pd.DataFrame(data_list, columns=rs.fields)
        df.rename(columns={'date':'Date', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col])

        # 保存为临时文件
        temp_file = f"temp_data_{stock_code}.csv"
        df.to_csv(temp_file, index=False)

        # ================= 3. 模型推理与逆向重构 =================
        loader = StockDataLoader(file_path=temp_file, sequence_length=20, train_split=0.8)
        _, _, X_test, y_test, test_base_close = loader.process_data()

        display_days = 60
        X_test_tensor = torch.tensor(X_test[-display_days:], dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test[-display_days:], dtype=torch.float32)
        base_close_recent = test_base_close[-display_days:]

        with torch.no_grad():
            predictions = model(X_test_tensor).numpy()

        pred_diff_real = loader.scaler_y.inverse_transform(predictions).flatten()
        y_test_diff_real = loader.scaler_y.inverse_transform(y_test_tensor.numpy()).flatten()

        pred_real_price = base_close_recent + pred_diff_real
        y_test_real_price = base_close_recent + y_test_diff_real

        if os.path.exists(temp_file):
            os.remove(temp_file)

        # ================= 4. 提取真实日历给前端 =================
        actual_dates = df['Date'].tolist()[-display_days:]
        if len(actual_dates) > 0:
            actual_dates[-1] = actual_dates[-1] + " (最新)"

        return jsonify({
            "status": "success",
            "dates": actual_dates,
            "real_prices": np.round(y_test_real_price, 2).tolist(),
            "predicted_prices": np.round(pred_real_price, 2).tolist()
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    print("====== 深度学习股价预测 Web 平台已启动 ======")
    app.run(debug=True, port=5000)