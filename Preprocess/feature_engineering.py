import os
import pandas as pd
import time
import numpy as np

def covert_time_to_seconds(txn_time):
    """
    將時間字串 (HH:MM:SS) 轉換為當天從 00:00:00 開始計算的總秒數。

    若輸入為無效值或空值 (NaN)，則返回 0。

    Args:
        txn_time (str): 格式為 'HH:MM:SS' 的時間字串。

    Returns:
        int: 轉換後的總秒數 (0 - 86399)。
    """
    try:
        if pd.isna(txn_time):
            return 0
        else:
            parts = txn_time.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        
    except:
        print(f"Error{txn_time}")
        return 0
    
def FeatureEngineering(txn):
    """
    執行特徵工程，從原始交易紀錄中提取帳戶的多維度特徵。

    針對每個帳戶計算匯出 (Send) 與匯入 (Receive) 的統計量，包含：
    - 金額統計 (總和、最大、最小、平均、標準差、次數)
    - 交易對象數量
    - 自身轉帳次數
    - 交易渠道與幣別的多樣性
    - 交易時間特徵 (平均時間、夜間交易比例)
    - 活躍天數
    - 是否為玉山銀行帳戶 (is_esun)

    Args:
        txn (pd.DataFrame): 原始交易紀錄資料表，需包含 `from_acct`, `to_acct`, `txn_amt` 等欄位。

    Returns:
        pd.DataFrame: 處理後的帳戶特徵資料表，每一列代表一個獨立帳戶及其特徵。
    """
    print("Feature Engineering...")

    # From accts info
    send_info = txn.groupby('from_acct')['txn_amt'].agg([
        ('total_send', 'sum'),
        ('max_send', 'max'),
        ('min_send', 'min'),
        ('avg_send', 'mean'),
        ('std_send', 'std'),
        ('send_count', 'count')
    ]).fillna(0)

    # Recv accts info
    recv_info = txn.groupby('to_acct')['txn_amt'].agg([
        ('total_recv', 'sum'),
        ('max_recv', 'max'),
        ('min_recv', 'min'),
        ('avg_recv', 'mean'),
        ('std_recv', 'std'),
        ('recv_count', 'count')
    ]).fillna(0)

    # Number of trading partners
    unique_send_to = txn.groupby('from_acct')['to_acct'].nunique().rename('number_of_recv_accts')
    unique_recv_from = txn.groupby('to_acct')['from_acct'].nunique().rename('number_of_send_accts')
    
    # Self-transfer account
    self_txn = txn[txn['is_self_txn'] == 'Y']
    self_send = self_txn.groupby('from_acct').size().rename('self_txn_count')

    # Channel usage statistics
    channel_diversity = txn.groupby('from_acct')['channel_type'].nunique().rename('channel_diversity')

    # Currency usage statistics
    currency_diversity = txn.groupby('from_acct')['currency_type'].nunique().rename('currency_diversity')

    # Time
    txn['time_seconds'] = txn['txn_time'].apply(covert_time_to_seconds)
    time_stats = txn.groupby('from_acct')['time_seconds'].agg([
        ('avg_time', 'mean'),
        ('std_time', 'std')
    ]).fillna(0)

    # Late-night trading ratio(22:00-06:00)
    txn['is_night'] = ((txn['time_seconds'] >= 79200) | (txn['time_seconds'] <= 21600)).astype(int)
    night_txn_ratio = txn.groupby('from_acct')['is_night'].mean().rename('night_txn_ratio')

    # Active days
    active_days = txn.groupby('from_acct')['txn_date'].nunique().rename('active_days')

    # 合併所有特徵（from_acct視角）
    df_from = pd.concat([
        send_info, unique_send_to, self_send, 
        channel_diversity, currency_diversity,
        time_stats, night_txn_ratio, active_days
    ], axis=1).fillna(0)
    
    # 合併所有特徵（to_acct視角）
    df_to = pd.concat([recv_info, unique_recv_from], axis=1).fillna(0)
    
    all_accts = pd.concat([
        df_from.reset_index().rename(columns={'from_acct': 'acct'}),
        df_to.reset_index().rename(columns={'to_acct': 'acct'})
    ]).groupby('acct').sum().fillna(0)
    
    # 添加 is_esun 特徵
    df_from_type = txn[['from_acct', 'from_acct_type']].drop_duplicates()
    df_to_type = txn[['to_acct', 'to_acct_type']].drop_duplicates()
    df_from_type.columns = ['acct', 'acct_type']
    df_to_type.columns = ['acct', 'acct_type']
    df_acct_type = pd.concat([df_from_type, df_to_type]).drop_duplicates().reset_index(drop=True)
    df_acct_type['is_esun'] = (df_acct_type['acct_type'] == 1).astype(int)
    
    all_accts = all_accts.merge(df_acct_type[['acct', 'is_esun']], left_index=True, right_on='acct', how='left') 
    all_accts['is_esun'] = all_accts['is_esun'].fillna(0)
    
    all_accts = all_accts.reset_index(drop=True) 
    cols = all_accts.columns.tolist()
    cols.insert(0, cols.pop(cols.index('acct')))
    all_accts = all_accts[cols]

    print(f"Total accounts: {len(all_accts)}, Features: {all_accts.shape[1]-1}")
    print("\n" + "="*60)
    
    return all_accts

    

if __name__ =="__main__":

    start = time.perf_counter()
    print("\n" + "="*60)
    root = "../Dataset/初賽資料/"

    # Output dir
    output_dir = "../feature/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.random.seed(42)

    # Load csv
    print("Load CSV...")
    txn = pd.read_csv(os.path.join(root, "acct_transaction.csv"))
    alert = pd.read_csv(os.path.join(root, "acct_alert.csv"))
    pred = pd.read_csv(os.path.join(root, "acct_predict.csv"))

    unique_from_accts = txn['from_acct'].nunique()
    all_unique_accts = set(txn['from_acct'].unique()) | set(txn['to_acct'].unique())
    total_unique_accts = len(all_unique_accts)

    print(f"Total Trade number: {len(txn)}, alert acct: {len(alert)}, pred acct: {len(pred)}")
    print(total_unique_accts)
    print("\n" + "="*60)

    #Feature Engineering
    features_data = FeatureEngineering(txn)
    
    # Output CSV
    output_data = os.path.join(output_dir, 'feature.csv')
    features_data.to_csv(output_data, index=False)

    end = time.perf_counter_ns()
    
    print(f"Feature Engineering Complete.\nCost Time:{(end-start)/1000000000:.2f} s")
