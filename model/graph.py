"""
Graph Construction
-------------------------
將特徵資料轉換為圖神經網路所需的圖結構資料

主要功能：
1. 建立節點 (Nodes)：每個帳戶視為一個節點，並賦予特徵工程產出的節點特徵。
2. 建立邊 (Edges)：根據交易紀錄建立有向邊，並提取交易金額、時間等邊特徵。
3. 生成標籤 (Labels)：標記已知的警示帳戶 (Alert Accounts)。
4. 儲存資料：將 PyTorch Geometric Data 物件與帳戶索引對照表儲存至 `graph/` 目錄。

Usage:
    python Preprocess/graph.py
"""

import os
import pandas as pd
import numpy as np
import torch
import time
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def LoadCSV(dir_path):
    """
    讀取原始資料集 CSV 檔案。

    Args:
        dir_path (str): 資料集所在的目錄路徑。

    Returns:
        tuple: (df_txn, df_alert, df_test)
            - df_txn (pd.DataFrame): 交易明細資料。
            - df_alert (pd.DataFrame): 警示帳戶名單。
            - df_test (pd.DataFrame): 需預測的測試集帳戶名單。
    """
    df_txn = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'))
    df_alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    df_test = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'))
    
    print(f"Load csv\n Total Trade number : {len(df_txn)}, alert acct: {len(df_alert)}, pred acct: {len(df_test)}")
    return df_txn, df_alert, df_test

def BuildGraph(df_txn, df_features, df_alert, df_test):
    """
    建構圖結構資料 (Graph Data Construction)。

    建立流程：
    1. 節點映射：建立帳號到索引 (Index) 的對照表。
    2. 邊的建構：根據交易紀錄建立來源節點到目標節點的邊 (Edge Index)。
    3. 邊特徵：提取交易金額、時間、是否轉給自己等特徵 (Edge Attributes)。
    4. 節點特徵：載入特徵工程後的帳戶特徵並進行標準化 (StandardScaler)。
    5. 標籤設定：標記 Alert 帳戶為 1，正常帳戶為 0。
    6. 資料分割：劃分訓練/驗證集 (Train/Val) 與測試集 (Test) 索引。

    Args:
        df_txn (pd.DataFrame): 交易明細資料。
        df_features (pd.DataFrame): 帳戶特徵資料。
        df_alert (pd.DataFrame): 警示帳戶資料。
        df_test (pd.DataFrame): 預測名單資料。

    Returns:
        tuple: (data, acct_to_idx)
            - data (torch_geometric.data.Data): PyG 圖資料物件，包含 x, edge_index, edge_attr, y, masks。
            - acct_to_idx (dict): 帳號對應到圖節點索引的字典。
    """
    print("Start Building Graph...")
    print("="*60)
    
    # Create account to index mapping
    all_accts = df_features['acct'].unique()
    acct_to_idx = {acct: idx for idx, acct in enumerate(all_accts)}
    # (*** Removed ***) idx_to_acct = {idx: acct for acct, idx in acct_to_idx.items()}
    
    print(f"Total accounts: {len(all_accts):,}") 
    
    # Vectorized edge construction
    print("Building transaction edges (vectorized)...") 
    df_txn_valid = df_txn[
        df_txn['from_acct'].isin(acct_to_idx) & 
        df_txn['to_acct'].isin(acct_to_idx)
    ].copy()
    
    df_txn_valid['from_idx'] = df_txn_valid['from_acct'].map(acct_to_idx)
    df_txn_valid['to_idx'] = df_txn_valid['to_acct'].map(acct_to_idx)
    
    edge_index = torch.tensor(
        df_txn_valid[['from_idx', 'to_idx']].values.T,
        dtype=torch.long
    )
    
    # Simplified edge features
    print("Extracting edge features...") 
    edge_amt = df_txn_valid['txn_amt'].fillna(0).astype(float).values
    edge_date = df_txn_valid['txn_date'].fillna(0).astype(int).values
    edge_self = df_txn_valid['is_self_txn'].map({'Y': 1.0, 'N': 0.0}).fillna(-1.0).values
    
    edge_attr = torch.tensor(
        np.column_stack([edge_amt, edge_date, edge_self]),
        dtype=torch.float
    )
    
    print(f"Total edges: {edge_index.shape[1]:,}") 
    
    # Node features (vectorized)
    print("Building node features...")
    
    # Ensure 'acct' (account ID) and 'index' (if it still exists) are not used as features
    feature_cols = [col for col in df_features.columns if col not in ['acct', 'index']]
    
    acct_feature_map = df_features.set_index('acct')[feature_cols].to_dict('index')
    
    node_features = np.zeros((len(all_accts), len(feature_cols)))
    for i, acct in enumerate(all_accts):
        if acct in acct_feature_map:
            node_features[i] = list(acct_feature_map[acct].values())
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Standardize features
    print("Standardizing features...") 
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(x), dtype=torch.float)
    
    # Labels (vectorized)
    print("Building labels and masks...") 
    alert_set = set(df_alert['acct'].values)
    y = torch.tensor([1 if acct in alert_set else 0 for acct in all_accts], dtype=torch.long)
    
    # (*** K-Fold Modification ***)
    # Do not create train_mask and val_mask here
    test_mask = torch.zeros(len(all_accts), dtype=torch.bool)
    
    # E.Sun accounts not in the test set are for training + validation
    esun_set = set(df_features[df_features['is_esun'] == 1]['acct'].values)
    test_set = set(df_test['acct'].values)
    
    train_val_indices = [
        acct_to_idx[acct] for acct in esun_set 
        if acct not in test_set and acct in acct_to_idx
    ]
    
    # Test set
    test_indices = [acct_to_idx[acct] for acct in test_set if acct in acct_to_idx]
    test_mask[test_indices] = True
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # (*** K-Fold Modification ***)
    # Store indices needed for K-Fold
    data.train_val_indices = torch.tensor(train_val_indices, dtype=torch.long)
    data.test_mask = test_mask # Test mask remains unchanged
    
    print("\n" + "="*60)
    print("(Finish) Build Graph")
    print(f"Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}")
    print(f"Train/Val candidates: {len(data.train_val_indices):,}, Test nodes: {data.test_mask.sum():,}")
    print(f"Train/Val alert ratio: {y[data.train_val_indices].float().mean():.2%}")
    print("="*60 + "\n")
    
    return data, acct_to_idx

# (*** New: Save function ***)
def save_graph_data(data, acct_to_idx, output_dir):
    """
    儲存處理後的圖資料與索引對照表。

    Args:
        data (torch_geometric.data.Data): 建構好的圖資料物件。
        acct_to_idx (dict): 帳號與索引對照表。
        output_dir (str): 輸出目錄路徑。

    Returns:
        None
    """
    print(f"Saving graph data to: {output_dir}")
    
    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save graph data
    graph_path = os.path.join(output_dir, 'graph_data.pt')
    torch.save(data, graph_path)
    
    # Save mapping dictionary (*** Changed to CSV ***)
    mapping_path = os.path.join(output_dir, 'acct_mappings.csv')
    
    # Convert acct_to_idx (dict) to DataFrame
    # Format: acct | idx
    df_mappings = pd.DataFrame(list(acct_to_idx.items()), columns=['acct', 'idx'])
    
    # Save as CSV
    df_mappings.to_csv(mapping_path, index=False)
    
    print(f"  - Graph data saved: {graph_path}")
    print(f"  - Account mapping saved: {mapping_path}")

if __name__ == "__main__":
    start = time.perf_counter_ns()

    root = "../"
    
    # Output
    output_dir = os.path.join(root, "graph")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.random.seed(42)
    torch.manual_seed(42)



    # Input
    print("Load CSV...")
    txn = pd.read_csv(os.path.join(root, "Dataset/初賽資料/acct_transaction.csv"))
    alert = pd.read_csv(os.path.join(root, "Dataset/初賽資料/acct_alert.csv"))
    pred = pd.read_csv(os.path.join(root, "Dataset/初賽資料/acct_predict.csv"))

    print("\n" + "="*60)

    # Load feature
    feature_path = os.path.join(root, 'feature/feature.csv')
    print(f"Loading precomputed features from {feature_path}...")
    try:
        features = pd.read_csv(feature_path)
        print(f"Loaded {len(features):,} account features.")
    except FileNotFoundError:
        print(f"Error: {feature_path} not found!")
        exit()
    except Exception as e:
        print(f"Error loading {feature_path}: {e}")
        exit()

    print("\n" + "="*60)
    
    # Build Graph
    data, acct_to_idx = BuildGraph(txn, features, alert, pred)
    save_graph_data(data, acct_to_idx, output_dir)

    end = time.perf_counter_ns()
    print(f"Feature Engineering Complete.\nCost Time:{(end-start)/1000000000:.2f} s")