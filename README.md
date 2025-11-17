# 2025 AICUP - Graph Neural Network for Fraud Detection

本專案為 2025 AICUP 比賽程式碼。我們使用 **GraphSAGE + GAT** 模型進行異常帳戶偵測，透過圖神經網路捕捉交易之間的關聯性。

## 1. 環境需求 (Requirements)
* **Python Version**: 3.8.0
* **OS**: Windows


### Installation
建議建立一個新的 Conda 環境以確保復現性：
```bash
conda create --name Final_AICUP python=3.8.0
conda activate Final_AICUP
pip install -r requirements.txt
```

## 2. Data Download

雲端下載連結: https://drive.google.com/drive/folders/1xAudFl5vnxj_HW-Bgmr6WuTVNWfk3wM9?usp=sharing

* Dataset

* feature

* graph

## 3. 專案結構 (Project Structure)

├── Preprocess/              [資料前處理]

: ├── feature_engineering.py
    
    └── graph.py

├── Model/                   [模型訓練與推論]

    ├── model.py             定義 GNN 模型架構 (GraphSAGE + GAT)

    ├── train.py             模型訓練

    ├── pred.py              模型推論

    └── analyze.py           結果分析腳本

├── Dataset/                 [資料集] 存放原始比賽資料

├── feature/                 [輸出] 存放 feature_engineering.py 產出的 csv

├── graph/                   [輸出] 存放 graph.py 產出的圖資料

├── result/                  [輸出] 存放訓練模型 (.pth) 與分析結果

├── requirements.txt         環境套件清單

└── README.md               專案說明文件

## 4. 檔案功能說明 (File Descriptions)

### Preprocess (前處理)
Preprocess/feature_engineering.py: 讀取 Dataset/ 資料，生成特徵並輸出至 feature/feature.csv。

Preprocess/graph.py: 讀取 feature.csv，轉換為圖結構資料，輸出至 graph/。

### Model (模型)
Model/model.py: 定義 GraphSAGE 與 GAT 的模型類別 (Class) 與架構細節。

Model/train.py: 載入圖資料進行訓練，模型權重將儲存於 result/ 下的日期資料夾。

Model/pred.py: 讀取訓練好的權重進行預測，產出 acct_pred.csv。

Model/analyze.py: 分析模型特徵重要性，輸出圖表至 result/

## 5. 執行步驟 (Usage Steps)

### 特徵工程，進入Preprocess fold
1.      python ./feature_engineering.py

### 圖資料建構，進入Preprocess fold
2.      python ./graph.py

### 模型訓練，進入model fold
3.      python ./train.py


### 產生預測，進入model fold
4.      python ./pred.py

### 結果分析 (optional)，進入model fold 
        python ./analyze.py

## 參考資料

模型
* https://arxiv.org/abs/2006.04637

AI 工具

程式碼修正
* gemini、claude

















