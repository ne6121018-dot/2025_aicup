import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
import time
from model import FraudGNN

# =============================================================================
# Load Pre-built Graph Data (Prediction Version)
# =============================================================================
def load_graph_data(processed_dir, raw_data_dir):
    """
    Load the pre-processed graph data and mappings for PREDICTION.
    - processed_dir: Directory containing graph_data.pt and acct_mappings.csv
    - raw_data_dir: Directory containing acct_predict.csv
    """
    print(f"Loading pre-processed data from: {processed_dir}")
    
    # 1. Load graph data
    graph_path = os.path.join(processed_dir, 'graph_data.pt')
    try:
        data = torch.load(graph_path)
    except FileNotFoundError:
        print(f"[Error] Graph file not found: {graph_path}")
        print("         Please run `graph.py` first to generate the graph data.")
        exit()
    
    # 2. Load account mappings from CSV
    mapping_path = os.path.join(processed_dir, 'acct_mappings.csv')
    try:
        df_mappings = pd.read_csv(mapping_path)
        # Re-create dictionaries
        acct_to_idx = {row['acct']: row['idx'] for _, row in df_mappings.iterrows()}
        idx_to_acct = {idx: acct for acct, idx in acct_to_idx.items()}
    except FileNotFoundError:
        print(f"[Error] Mapping file not found: {mapping_path}")
        print("         Please run `graph.py` first to generate the mappings.")
        exit()

    # 3. (*** Added ***) Load the test (prediction) file
    test_file_path = os.path.join(raw_data_dir, 'acct_predict.csv')
    try:
        df_test = pd.read_csv(test_file_path)
    except FileNotFoundError:
        print(f"[Error] Test file not found: {test_file_path}")
        print(f"         Please ensure 'acct_predict.csv' is in {raw_data_dir}")
        exit()
        
    print("Successfully loaded graph data, mappings, and test file.")
    return data, acct_to_idx, idx_to_acct, df_test

# =============================================================================
# Prediction Functions
# =============================================================================

def Predict(model, data, idx_to_acct):
    """
    Predict on the test set
    - Returns df_pred (with probabilities) for ensembling
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        prob = F.softmax(out, dim=1)[:, 1]  # Probability of being an alert (class 1)
    
    # Extract test set results
    test_indices = torch.where(data.test_mask)[0].cpu().numpy()
    test_accts = [idx_to_acct[idx] for idx in test_indices]
    test_probs = prob[data.test_mask].cpu().numpy()
    
    df_pred = pd.DataFrame({
        'acct': test_accts,
        'probability': test_probs
    })
    
    return df_pred


def OutputCSV(path, df_test, df_pred_ensembled):
    """
    Output prediction results
    - df_pred_ensembled contains the final ensembled probabilities and labels
    """
    df_out = df_test[['acct']].merge(df_pred_ensembled[['acct', 'label']], on='acct', how='left')
    df_out['label'] = df_out['label'].fillna(0).astype(int)
    df_out.to_csv(path, index=False)
    print(f"Final Submission File Saved: {path}\n")
    
    # Also save probabilities for manual threshold tuning
    prob_path = path.replace('.csv', '_proba.csv')
    df_prob_out = df_test[['acct']].merge(df_pred_ensembled[['acct', 'probability']], on='acct', how='left')
    df_prob_out['probability'] = df_prob_out['probability'].fillna(0.0)
    df_prob_out.to_csv(prob_path, index=False)
    print(f"Probability File Saved: {prob_path}\n")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("E.Sun AI Challenge - GNN Prediction Script (K-Fold Ensembling)")
    print("="*60)
    
    # --- (*** Path Settings ***) ---
    # (Must match the paths used in train.py)
    # 1. Directory for original CSVs
    data_dir = "../Dataset/初賽資料/" 
    # 2. Directory where graph.py/train.py saved outputs
    processed_dir = "../result/115_1529_kfold"

    graph_dir = "../graph"
    # --- (*** End Path Settings ***) ---
    
    output_dir = processed_dir # Save results in the same folder
    
    # Set random seeds (for consistency, although not strictly needed for predict)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    
    # 1. Load Pre-built Graph Data
    # (*** Modified ***) Now receives df_test
    data, acct_to_idx, idx_to_acct, df_test = load_graph_data(graph_dir, data_dir)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # (*** K-Fold Prediction Loop ***)
    N_SPLITS = 5 # (Must match N_SPLITS from train.py)
    
    all_test_probs = [] # To store test set prediction probabilities from all 5 models
    
    # (*** Model Hyperparameters ***)
    # (Must match model_hparams from train.py)
    model_hparams = {
        'in_channels': data.x.shape[1],
        'hidden_channels': 80,
        'num_layers': 4,
        'dropout': 0.4
    }
    
    print("Starting K-Fold Ensemble Prediction...")
    for fold in range(1, N_SPLITS + 1):
        print(f"--- Loading and Predicting with Fold {fold}/{N_SPLITS} ---")
        
        # 1. Define model path
        model_save_path = os.path.join(output_dir, f'best_model_fold_{fold}.pt')
        
        try:
            # 2. Create model instance
            model = FraudGNN(**model_hparams)
            # 3. Load trained weights
            model.load_state_dict(torch.load(model_save_path))
        except FileNotFoundError:
            print(f"[Error] Model file not found: {model_save_path}")
            print("         Please run `train.py` first to generate the models.")
            exit()
        except RuntimeError as e:
            print(f"[Error] Failed to load model weights: {e}")
            print("         Ensure `model_hparams` in this script match `train.py`.")
            exit()
            
        # 4. Predict probabilities
        df_pred_fold = Predict(model, data, idx_to_acct)
        all_test_probs.append(df_pred_fold['probability'].values)
        
    print("K-Fold Prediction Complete.")

    # (*** Model Ensembling ***)
    print("\n" + "="*60)
    print("Ensembling predictions from all 5 models...")
    # 'all_test_probs' is a list of arrays, shape (K, num_test_samples)
    # We average them along the K dimension (axis=0)
    avg_test_prob = np.mean(all_test_probs, axis=0)
    
    # Create final prediction DataFrame
    test_accts = [idx_to_acct[idx] for idx in torch.where(data.test_mask)[0].cpu().numpy()]
    
    # (*** Threshold ***)
    # This is the final threshold for classification.
    # You can tune this (e.g., 0.5, 0.75) to optimize the Public LB score.
    FINAL_THRESHOLD = 0.85
    print(f"Using final threshold: {FINAL_THRESHOLD}")
    
    df_pred_ensembled = pd.DataFrame({
        'acct': test_accts,
        'probability': avg_test_prob,
        'label': (avg_test_prob > FINAL_THRESHOLD).astype(int)
    })
    
    print(f"Ensembled predictions (Alert=1): {df_pred_ensembled['label'].sum()} / {len(df_pred_ensembled)}")
    
    # 8. Output final ensembled results
    result_path = os.path.join(output_dir, f'result_gnn_ensembled_{FINAL_THRESHOLD}.csv')
    OutputCSV(result_path, df_test, df_pred_ensembled)
    
    print("="*60)
    print(f"\nAll output files are located in: {output_dir}/")
    print("- result_gnn_ensembled.csv (*** Your final submission file ***)")
    print("- result_gnn_ensembled_proba.csv (Contains probabilities for threshold tuning)")
    print("="*60)
