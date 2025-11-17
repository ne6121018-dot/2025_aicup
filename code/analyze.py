import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import GNNExplainer 
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import FraudGNN

# =============================================================================
# Helper Functions
# (Only FraudGNN and PlotFeatureImportance are needed)
# =============================================================================

def PlotFeatureImportance(importance_df, save_path):
    """
    Plot GNNExplainer's global feature importance
    """
    print(f"Saving GNN feature importance plot to {save_path}")
    # Display Top 20
    top_20 = importance_df.head(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(top_20['feature'], top_20['importance'], color='steelblue')
    plt.xlabel("Average Importance (GNNExplainer)")
    plt.ylabel("Feature")
    plt.title("GNN Global Feature Importance (Top 20)")
    plt.gca().invert_yaxis() # Show most important at top
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# (*** NEW: Simplified data loader ***)
def load_analysis_data(kfold_dir):
    """
    Load all necessary pre-built data for analysis.
    - kfold_dir: Directory where graph.py and train.py saved their outputs.
    """
    print(f"Loading pre-built data from: {kfold_dir}")
    
    # 1. Load graph data
    graph_path = '../graph/graph_data.pt'
    try:
        data = torch.load(graph_path)
    except FileNotFoundError:
        print(f"[Error] Graph file not found: {graph_path}")
        exit()

    # 2. Load account mappings
    mapping_path = '../graph/acct_mappings.csv'
    try:
        df_mappings = pd.read_csv(mapping_path)
        acct_to_idx = {row['acct']: row['idx'] for _, row in df_mappings.iterrows()}
    except FileNotFoundError:
        print(f"[Error] Mapping file not found: {mapping_path}")
        exit()

    # 3. Load feature columns list
    feature_csv_path = '../feature/feature.csv'
    try:
        df_features = pd.read_csv(feature_csv_path)
        # (*** Bug Fix ***)
        # Ensure 'acct' (account ID) and 'index' (if it exists) are NOT used as features
        feature_cols = [col for col in df_features.columns if col not in ['acct', 'index']]
        print(f"✓ Loaded feature names from {feature_csv_path}")
    except FileNotFoundError:
        print(f"[Error] Feature file not found: {feature_csv_path}")
        print("         Please ensure feature.csv is in the kfold directory.")
        exit()
        
    # 4. Load ensembled predictions
    pred_file = os.path.join(kfold_dir, 'result_gnn_ensembled.csv')
    try:
        df_pred_ensembled = pd.read_csv(pred_file)
    except FileNotFoundError:
        print(f"[Error] Prediction file not found: {pred_file}")
        exit()
        
    print("✓ All necessary data loaded.")
    return data, acct_to_idx, feature_cols, df_pred_ensembled


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GNNExplainer (SHAP) Analysis Script")
    print("="*60)

    # (*** Settings ***)
    # *** (!!IMPORTANT!!) This must match the output folder of train.py ***
    kfold_output_dir = "../result/114_06_kfold" 
    analysis_output_dir = kfold_output_dir
    
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)
    
    # *** (!!IMPORTANT!!) These MUST match the params in train.py ***
    model_hparams = {
        'in_channels': -1, # Will be auto-filled
        'hidden_channels': 80,
        'num_layers': 4,
        'dropout': 0.4
    }
    
    # 1. Load all pre-built data
    # (*** Modified ***) No longer loads raw data or re-builds graph
    print("Step 1: Loading all pre-built data...")
    data, acct_to_idx, feature_cols, df_pred_ensembled = load_analysis_data(
        kfold_output_dir
    )
    
    # Update in_channels
    model_hparams['in_channels'] = data.x.shape[1]
    
    # 2. Load a representative model (Fold 1)
    print("\nStep 2: Loading Fold 1 model as representative...")
    model_to_explain = FraudGNN(**model_hparams)
    model_path = os.path.join(kfold_output_dir, 'best_model_fold_1.pt')
    try:
        model_to_explain.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"[Error] Model file not found: {model_path}")
        print(f"         Please run `train.py` first to generate models.")
        exit()
    except RuntimeError as e:
        print(f"[Error] Failed to load model: {e}")
        print("         Please ensure `model_hparams` in this script exactly match `train.py`.")
        exit()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_to_explain = model_to_explain.to(device)
    model_to_explain.eval()

    
    # (*** GNNExplainer/SHAP Analysis ***)
    print("\n" + "="*60)
    print("Starting GNNExplainer (SHAP) Analysis...")
    try:
        # 1. Setup GNNExplainer (v2.0.4 API)
        explainer = GNNExplainer(
            model=model_to_explain,
            epochs=100,
            log=False # Disable verbose logging
        )
        
        # 2. Find target nodes to explain (nodes predicted as 1 in the test set)
        nodes_to_explain_accts = df_pred_ensembled[df_pred_ensembled['label'] == 1]['acct']
        nodes_to_explain_indices = [acct_to_idx[acct] for acct in nodes_to_explain_accts if acct in acct_to_idx]
        
        if len(nodes_to_explain_indices) == 0:
            print("  - [Warning] Ensembled model did not predict any test nodes as 1. Cannot run SHAP/Explainer analysis.")
        else:
            # For speed, we sample max 50 nodes to explain
            if len(nodes_to_explain_indices) > 50:
                print(f"  - Sampling 50 (out of {len(nodes_to_explain_indices)}) predicted-as-1 nodes for analysis...")
                nodes_to_explain_indices = np.random.choice(nodes_to_explain_indices, 50, replace=False)
            else:
                print(f"  - Analyzing {len(nodes_to_explain_indices)} nodes predicted as 1...")

            data_gpu = data.to(device)
            all_feat_masks = []

            for node_idx in tqdm(nodes_to_explain_indices, desc="GNNExplainer Analyzing"):
                node_feat_mask, edge_mask = explainer.explain_node(
                    node_idx=int(node_idx),
                    x=data_gpu.x, 
                    edge_index=data_gpu.edge_index
                )
                all_feat_masks.append(node_feat_mask.cpu().numpy())
            
            # 3. Aggregate global feature importance
            if len(all_feat_masks) > 0:
                global_feat_importance = np.mean(all_feat_masks, axis=0).flatten() # Flatten
                
                # 4. Create DataFrame and sort
                if len(global_feat_importance) == len(feature_cols):
                    importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': global_feat_importance
                    }).sort_values(by='importance', ascending=False)
                    
                    print("\n--- GNN Global Feature Importance (Top 10) ---")
                    print(importance_df.head(10))
                    
                    # 5. Save report and chart
                    plot_path = os.path.join(analysis_output_dir, 'gnn_feature_importance.png')
                    PlotFeatureImportance(importance_df, plot_path)
                    
                    csv_path = os.path.join(analysis_output_dir, 'gnn_feature_importance.csv')
                    importance_df.to_csv(csv_path, index=False)
                    print(f"Save GNN feature importance csv to {csv_path}")
                else:
                    print(f"  [Error] GNNExplainer feature mask dim ({len(global_feat_importance)}) does not match feature list dim ({len(feature_cols)}).")

            else:
                 print("  - [Warning] No GNNExplainer results were generated.")

    except ImportError:
        print("\n[Error] Could not find 'torch_geometric.nn.GNNExplainer'.")
    except Exception as e:
        print(f"\n[Error] GNNExplainer analysis failed: {e}")

    print("GNNExplainer analysis finished.")
    print(f"Analysis results (if generated) are in: {analysis_output_dir}/")
    print("="*60)

