# 基于 CNN-LSTM-Attention 的智能股价预测系统

## 📖 项目简介
本项目是一个基于深度学习混合模型（CNN-LSTM-Attention）的动态股价预测系统。系统通过一维卷积神经网络（1D-CNN）提取股票 K 线数据的局部微观特征，结合长短期记忆网络（LSTM）捕捉长程时序依赖，并引入注意力机制（Attention）动态分配关键时间步权重，有效提升了对非平稳金融时序数据的预测精度。

系统采用前后端分离架构，后端基于 Flask 提供深度学习在线推理引擎，前端基于 Vue3 与 ECharts 实现高保真数据可视化。

## 🛠️ 技术栈
- **算法模型**：1D-CNN, LSTM, Attention 机制 (基于 TensorFlow/PyTorch)
- **数据处理**：Pandas, NumPy, 一阶差分重构
- **后端框架**：Python 3.x, Flask, Baostock 金融接口
- **前端框架**：Vue3, ECharts 5.0, Axios

## 📂 核心项目结构
```text
Stock-Prediction-System/
├── backend/                 # Flask 后端服务
│   ├── app.py               # 后端主入口，定义 API 路由
│   ├── model_engine.py      # CNN-LSTM-Attention 模型推理逻辑
│   ├── data_fetcher.py      # Baostock 数据拉取与一阶差分处理模块
│   ├── requirements.txt     # Python 依赖清单
│   └── cache/               # 本地 CSV 降级缓存目录（断网容错）
└── frontend/                # Vue3 前端展示
    ├── src/
    │   ├── components/      # ECharts 可视化组件
    │   ├── api/             # Axios 请求封装
    │   └── App.vue          # 主页面
    ├── package.json         # Node 依赖清单
    └── vite.config.js       # 构建配置


🚀 快速开始
1. 后端环境配置 (Flask + 算法引擎)
请确保你的机器上已安装 Python 3.8+。

Bash
# 进入后端目录
cd backend

# 创建并激活虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows 用户使用 venv\Scripts\activate

# 安装依赖库
pip install -r requirements.txt

# 启动 Flask 服务（默认运行在 http://localhost:5000）
python app.py
2. 前端环境配置 (Vue3)
请确保你的机器上已安装 Node.js (推荐 v16+)。

Bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器（默认运行在 http://localhost:3000 或 5173）
npm run dev
💡 使用指南
前后端服务均启动后，在浏览器中访问前端地址（如 http://localhost:5173）。

在系统界面的“输入股票代码”搜索框中，输入标准的 A 股代码（例如：sh.600519）。

点击“预测”按钮，系统将自动执行以下全链路操作：

触发后台 Baostock 接口动态拉取最新历史数据。

执行数据清洗、一阶差分与时间窗口动态锚定。

张量多维前向传播进入 CNN-LSTM-Attention 模型进行推理。

对输出张量进行差分逆向还原与序列化。

最终结果将在下方的 ECharts 面板中以双折线图（真实值 vs 预测值）呈现，并附带 MAE/RMSE 等评估指标。

🛡️ 容错机制说明
本系统内置了智能降级策略。当 Baostock API 因网络原因请求超时或失败时，系统将自动触发 alt 容错分支，切换至读取 backend/cache/ 目录下的本地 CSV 历史快照，确保推理引擎的可用性与系统的工业级稳定性。


***
