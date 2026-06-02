from flask import Flask, render_template, jsonify, request
import torch
import numpy as np
import pandas as pd
import os
import baostock as bs
from datetime import datetime, timedelta
import concurrent.futures  # 引入并发模块，用于控制网络请求超时

# 导入算法模块
from data_loader import StockDataLoader
from advanced_model import CNN_LSTM_Attention

app = Flask(__name__)

# ================= 1. 全局挂载 AI 模型 =================
print("正在将 CNN-LSTM-Attention 模型载入服务器内存...")
model = CNN_LSTM_Attention(input_size=8, cnn_out_channels=16, hidden_size=64, num_layers=2, dropout_rate=0.2)
# 加载模型权重
model.load_state_dict(torch.load('cnn_lstm_attn_model.pth', weights_only=True))
model.eval()
print("模型加载完毕，AI 引擎已就绪！")

@app.route('/')
def home():
    return render_template('index.html')

# ================= 独立封装的 Baostock 拉取函数 =================
def fetch_baostock_data(bs_code, start_date, end_date):
    """
    此函数将在独立线程中运行，方便主程序随时“掐断”它
    """
    lg = bs.login()
    if lg.error_code != '0':
        bs.logout()
        raise Exception(f"登录被拒绝: {lg.error_msg}")

    rs = bs.query_history_k_data_plus(bs_code,
        "date,open,high,low,close,volume",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2")

    data_list = []
    # 注意这里必须用 and，防止 WinError 10057
    while (rs.error_code == '0') and rs.next():
        data_list.append(rs.get_row_data())

    bs.logout()

    if not data_list:
        raise Exception("Baostock 服务器返回了空数据，可能正在维护或限流。")

    # 格式化为 DataFrame 返回
    df = pd.DataFrame(data_list, columns=rs.fields)
    return df


# ================= 2. 核心预测 API =================
@app.route('/api/predict', methods=['GET'])
def predict_api():
    try:
        stock_code = request.args.get('code', '600519')
        pure_code = stock_code.split('.')[-1] if '.' in stock_code else stock_code
        print(f"\n---> 收到前端请求，准备预测股票：{pure_code}")

        # 智能拼装 Baostock 需要的前缀格式
        if pure_code.startswith('6'):
            bs_code = f"sh.{pure_code}"
        elif pure_code.startswith('0') or pure_code.startswith('3'):
            bs_code = f"sz.{pure_code}"
        else:
            bs_code = f"sh.{pure_code}"

        # 仅保留传给 DataLoader 的临时文件，彻底移除 cache_file 变量
        temp_file = f"temp_data_{pure_code}.csv"
        
        # Baostock 需要的日期格式是 YYYY-MM-DD
        today_bs = datetime.now().strftime('%Y-%m-%d')
        two_years_ago_bs = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        df = pd.DataFrame()

        # ================= 纯实时网络拉取 =================
        print(f"正在尝试连接 Baostock 实时拉取 {bs_code} 最新数据...")
        try:
            # 开启线程池，强制设置最大等待时间为 15 秒（放宽超时，防止网络波动）
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_baostock_data, bs_code, two_years_ago_bs, today_bs)
                df_bs = future.result(timeout=15)  

            if not df_bs.empty:
                df_bs.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                df = df_bs[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col])
                
                # 写入供 DataLoader 读取的临时中间文件
                df.to_csv(temp_file, index=False)
                print(f"✅ 网络拉取成功，临时数据流已就绪！")
            else:
                return jsonify({"status": "error", "message": f"Baostock 未能返回有效数据，标的 {pure_code} 此时段可能停牌或不存在。"})
                
        except concurrent.futures.TimeoutError:
            print("❌ Baostock 响应超时，已强制掐断连接！")
            return jsonify({"status": "error", "message": "Baostock 实时数据接口响应超时，请稍后刷新重试。"})
        except Exception as e:
            print(f"❌ Baostock 拉取发生异常: {str(e)}")
            return jsonify({"status": "error", "message": f"数据拉取失败: {str(e)}"})

        # ================= 3. 模型推理与逆向重构 =================
        print("数据装载完毕，开始执行 CNN-LSTM-Attention 模型推理...")
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

        # 推理完成，立即销毁临时生成的转换文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # ================= 4. 提取真实日历给前端 =================
        actual_dates = df['Date'].tolist()[-display_days:]
        if len(actual_dates) > 0:
            actual_dates[-1] = actual_dates[-1] + " (最新)"

        print("✅ 推理完成，即将把 JSON 数据发回前端渲染！")
        return jsonify({
            "status": "success",
            "dates": actual_dates,
            "real_prices": np.round(y_test_real_price, 2).tolist(),
            "predicted_prices": np.round(pred_real_price, 2).tolist()
        })

    except Exception as e:
        print(f"❌ 后端发生致命错误: {str(e)}")
        # 兜底确保哪怕后续步骤报错，也会清理掉 temp_file
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    print("====== 深度学习股价预测 Web 平台已启动 ======")
    app.run(debug=True, port=5000)