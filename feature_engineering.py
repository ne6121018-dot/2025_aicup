import os
import pandas as pd
import time
import numpy as np

def covert_time_to_seconds(txn_time):
    # Format HH:MM:SS
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

    '''
    pred_accts = set(alert['acct'].unique())

    missing_accts = pred_accts - all_unique_accts

    if len(missing_accts) == 0:
        print("`acct_predict.csv` 中的帳戶都存在於 `acct_transaction.csv` 中。")
    else:
        print(f"有 {len(missing_accts)} 個 `acct_predict.csv` 中的帳戶不存在於 `acct_transaction.csv` 中！")
        print(f"  缺少範例帳戶: {list(missing_accts)[:5]}...")
    print("="*60 + "\n")
    '''

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
