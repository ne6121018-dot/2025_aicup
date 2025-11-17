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

## Data Download

雲端下載連結: https://drive.google.com/drive/folders/1xAudFl5vnxj_HW-Bgmr6WuTVNWfk3wM9?usp=sharing

* Dataset

* feature

* graph

### feature_engineering.py
生成特徵並輸出於feature/feature.csv

### graph.py
將生成的feature.csv產出graph_data.pt與acct_mappings.csv

### train.py
訓練模型，並且生成於當天日期的fold中，模型參數在此調整

### pred.py
訓練完的model預測acct_pred.csv

### analyze.py
分析特徵的重要性，輸出於每個訓練結果的資料夾中，gnn_feature_importance.csv與gnn_feature_importance.png

## Usage Steps
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

## 2. 參考資料

模型
* https://arxiv.org/abs/2006.04637

AI 工具

程式碼修正
* gemini、claude




