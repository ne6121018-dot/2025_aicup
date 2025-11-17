"""
Model Training Module
---------------------
本模組負責訓練 FraudGNN 模型。

主要功能：
1. 載入圖資料：讀取 `graph/graph_data.pt`。
2. K-Fold 交叉驗證：使用 StratifiedKFold 進行 5 折交叉驗證，確保訓練穩定性。
3. 模型訓練：
   - 使用 AdamW 優化器。
   - 應用 Class Weights 處理類別不平衡問題。
   - 支援混合精度訓練 (Mixed Precision Training) 以加速計算。
   - 實作 Early Stopping 機制防止過擬合。
4. 訓練記錄：儲存每個 Fold 的最佳模型權重 (.pt) 與訓練歷程圖表。

Usage:
    python model/train.py
"""
import time
import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from model import FraudGNN
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import logging
import json

def load_graph_data(grath_path):
    """
    載入預處理好的圖資料與映射表。

    Args:
        graph_dir_path (str): 包含 `graph_data.pt` 與 `acct_mappings.csv` 的目錄路徑。

    Returns:
        tuple: (data, acct_to_idx)
            - data: PyTorch Geometric Data 物件。
            - acct_to_idx: 帳號對應索引的字典。
    """
    logging.info(f"Loading pre-processed data from: {grath_path}")
    
    # 1. Load graph data
    graph_path = os.path.join(grath_path, 'graph_data.pt')
    try:
        data = torch.load(graph_path)
    except FileNotFoundError:
        logging.error(f"[Error] Graph file not found: {graph_path}")
        exit()
    
    # 2. Load account mappings from CSV
    mapping_path = os.path.join(grath_path, 'acct_mappings.csv')
    try:
        df_mappings = pd.read_csv(mapping_path)
        acct_to_idx = {row['acct']: row['idx'] for _, row in df_mappings.iterrows()}
    except FileNotFoundError:
        logging.error(f"[Error] Mapping file not found: {mapping_path}")
        exit()

    logging.info("Successfully loaded graph data and mappings.")
    return data, acct_to_idx

def TrainModel(model, data, model_save_path, fold_num=0, epochs=200, lr=0.001, weight_decay=5e-4): # <-- (*** tqdm added ***) Added fold_num
    """
    訓練 GNN 模型 (單一 Fold)。

    執行流程：
    1. 將模型與資料移至 GPU。
    2. 計算類別權重 (Class Weights) 以處理樣本不平衡。
    3. 進入訓練迴圈 (Training Loop)：
       - Forward Pass -> Loss 計算 -> Backward Pass -> Optimizer Step
    4. 進入驗證迴圈 (Validation Loop)：
       - 計算 Val Loss 與 F1-Score。
    5. Early Stopping 檢查：若 Val Loss 未改善則計數，超過耐心值則停止。
    6. 儲存最佳模型。

    Args:
        model (FraudGNN): 初始化的模型實例。
        data (Data): 包含 train_mask/val_mask 的圖資料。
        model_save_path (str): 模型權重儲存路徑。
        fold_num (int): 當前 Fold 編號 (用於顯示)。
        epochs (int): 最大訓練回合數。
        lr (float): 學習率。
        weight_decay (float): 權重衰減 (L2 Regularization)。

    Returns:
        tuple: (model, history, best_f1)
            - model: 訓練完成的模型。
            - history: 包含 loss, acc, f1 等歷程的字典。
            - best_f1: 該 Fold 的最佳 F1 分數。
    """
    # GPU setup
    device = torch.device('cuda')
    
    torch.cuda.empty_cache()
    
    model = model.to(device)
    data = data.to(device)
    
    # Mixed precision training setup
    scaler = torch.cuda.amp.GradScaler()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Handle class imbalance (*** CRITICAL ***)
    train_labels = data.y[data.train_mask]
    
    # (*** Fix ***) Avoid NaN if (train_labels == 1).sum() is 0
    if (train_labels == 1).sum() == 0:
        logging.warning("  [Warning] No positive samples (Alert=1) in this training fold!")
        class_weights = torch.tensor([1.0, 1.0]).to(device)
    else:
        class_weights = torch.tensor([
            1.0 / (train_labels == 0).sum(),
            1.0 / (train_labels == 1).sum()
        ]).to(device)
        class_weights = class_weights / class_weights.sum() * 2
    
    best_f1 = 0.0 
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    # Training history log
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [],
        'val_acc': [], 'precision': [], 'recall': [], 'f1': []
    }
    
    # (*** tqdm added ***) Add progress bar for epochs
    logging.info(f"  Training Fold {fold_num}...")
    pbar = tqdm(range(epochs), desc=f"  Epoch 0 | Val F1: 0.0000", leave=False)
    
    for epoch in pbar:
        # ========== Training Phase ==========
        model.train()
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            out = model(data.x, data.edge_index)
            train_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # ========== Evaluation Phase ==========
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            # Training metrics
            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
            
            # Validation loss (*** CRITICAL ***)
            # Ensure val_mask is not empty
            if data.val_mask.sum() == 0:
                logging.warning("  [Warning] Validation set for this fold is empty!")
                history['train_loss'].append(train_loss.item())
                history['val_loss'].append(0)
                history['train_acc'].append(train_acc.item())
                history['val_acc'].append(0)
                history['precision'].append(0)
                history['recall'].append(0)
                history['f1'].append(0)
                continue
                
            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
            
            # Validation metrics
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
            
            # Calculate precision, recall, and F1
            val_pred = pred[data.val_mask]
            val_true = data.y[data.val_mask]
            tp = ((val_pred == 1) & (val_true == 1)).sum().item()
            fp = ((val_pred == 1) & (val_true == 0)).sum().item()
            fn = ((val_pred == 0) & (val_true == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # print(f"Epoch:{epoch}/{epochs}, Training loss: {train_loss}, val loss: {val_loss}, f1: {f1}")
        
        # Log history
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_acc'].append(train_acc.item())
        history['val_acc'].append(val_acc.item())
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        
        # (*** K-Fold Modification ***)
        # Early stopping (based on F1 score)
        # if f1 > best_f1:
        if val_loss.item() < best_val_loss:
            best_f1 = f1
            best_val_loss = val_loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path) # Save to this fold's specific path
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logging.info(f'  - Early stopping at epoch {epoch+1}')
            pbar.close() # (*** tqdm added ***) Close pbar on early stop
            break
            
        # (*** tqdm added ***) Update pbar description
        if epoch % 10 == 0 or epoch == epochs - 1:
            pbar.set_description(f"  Epoch {epoch+1} | Val F1: {f1:.4f} (Best: {best_val_loss:.4f})")
    
    if epoch == epochs - 1:
        pbar.close() # (*** tqdm added ***) Close pbar on completion

    # (*** K-Fold Modification ***)
    # Load the best F1 model for this fold
    if os.path.exists(model_save_path): # Check if model was ever saved (if f1 > 0)
        model.load_state_dict(torch.load(model_save_path))
    else:
        logging.warning(f"  [Warning] No model saved for this fold (best F1 was {best_val_loss:.4f}).")

    
    logging.info(f"  - (Finish) Fold Training")
    logging.info(f"  - Best validation Loss: {best_val_loss:.4f} (F1 at this point: {best_f1:.4f})")
    
    return model, history, best_f1

def PlotTrainingHistory(history, save_path='training_history.png'):
    """
    繪製訓練歷程圖表並存檔。
    包含 Loss, Accuracy, Precision/Recall, F1 Score 四張子圖。
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GNN Training History (Fold Snapshot)', fontsize=16, fontweight='bold')
    
    # 1. Loss Curve
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy Curve
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision & Recall
    axes[1, 0].plot(history['precision'], label='Precision', linewidth=2)
    axes[1, 0].plot(history['recall'], label='Recall', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. F1 Score
    axes[1, 1].plot(history['f1'], label='F1 Score', linewidth=2, color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Save Graph: {save_path}")
    plt.close()

def SaveTrainingHistory(history, save_path='training_history.csv'):
    """將訓練數據儲存為 CSV 檔案"""
    df = pd.DataFrame.from_dict(history, orient='index').transpose()
    df.to_csv(save_path, index=False)
    logging.info(f"Training History saved to: {save_path}")

if __name__ == "__main__":
    t_format = time.localtime(time.time())

    # Input
    root = "../graph"

    # Output
    output_dir = f'../result/{t_format.tm_mon}{t_format.tm_mday}_{t_format.tm_hour}{t_format.tm_min}_kfold'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # log
    log_path = os.path.join(output_dir, 'log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Output directory set to: {output_dir}")

    logging.info(f"\nPyTorch Version: {torch.__version__}")
    logging.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA Version: {torch.version.cuda}")
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logging.info("="*60 + "\n")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    # Load Graph
    data, acct_to_idx = load_graph_data(root)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # (*** K-Fold 迴圈 ***)
    N_SPLITS = 5 

    # 使用 StratifiedKFold 確保每一折中的警示 (y=1) 比例盡可能一致
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    all_train_val_indices = data.train_val_indices.numpy()
    y_for_stratify = data.y[data.train_val_indices].numpy()

    fold_best_f1s = []

    fold_iterator = enumerate(skf.split(all_train_val_indices, y_for_stratify))
    
    for fold, (train_idx, val_idx) in fold_iterator:
        start_fold = time.perf_counter()
        
        # 1. Create train_mask and val_mask for this fold
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[all_train_val_indices[train_idx]] = True
        data.val_mask[all_train_val_indices[val_idx]] = True
        
        logging.info(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        logging.info(f"Train samples: {data.train_mask.sum()}, Val samples: {data.val_mask.sum()}")
        logging.info(f"Train Alert(1) count: {data.y[data.train_mask].sum()}")
        logging.info(f"Val Alert(1) count: {data.y[data.val_mask].sum()}")
        
        # 2. Create model (*** Re-initialize for each fold ***)
        logging.info("Creating GNN model...")
        logging.info(f"Feature num: {data.x.shape[1]}")
        model_hparams = {
            'in_channels': data.x.shape[1],
            'hidden_channels': 80,
            'num_layers': 4,
            'dropout': 0.4
        }
        model = FraudGNN(**model_hparams)
        logging.info(f"Model Hyperparameters: \n{json.dumps(model_hparams, indent=2)}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        
        # 3. Train model
        model_save_path = os.path.join(output_dir, f'best_model_fold_{fold+1}.pt')
        train_hparams = {
            'epochs': 400, 
            'lr': 0.008, 
            'weight_decay': 5e-4 #  5e-4
        }
        logging.info(f"Training Hyperparameters: \n{json.dumps(train_hparams, indent=2)}")
        
        model, history, best_f1 = TrainModel(
            model, data, model_save_path, 
            fold_num=fold+1, # (*** tqdm added ***) Pass fold number
            **train_hparams
        )

        
        fold_best_f1s.append(best_f1)
        logging.info(f"Fold {fold+1} F1 at Best Val Loss: {best_f1:.4f}")
        
        # 4. (*** Removed ***) Prediction on test set is moved to predict.py
        
        # Plot training history for this fold
        plot_path = os.path.join(output_dir, f'training_history_fold_{fold+1}.png')
        PlotTrainingHistory(history, save_path=plot_path)

        # (*** Added ***) Save training history CSV for this fold
        csv_path = os.path.join(output_dir, f'training_history_fold_{fold+1}.csv')
        SaveTrainingHistory(history, save_path=csv_path)

        end_fold = time.perf_counter()

        logging.info(f"Fold {fold+1} Time: {end_fold-start_fold:.2f}s")

        # (*** K-Fold Summary ***)
    logging.info("\n" + "="*60)
    logging.info("K-Fold Training Complete")
    avg_f1 = np.mean(fold_best_f1s)
    std_f1 = np.std(fold_best_f1s)
    logging.info(f"All Folds 'F1 @ Best Loss': {[round(f, 4) for f in fold_best_f1s]}")
    logging.info(f"Average 'F1 @ Best Loss': {avg_f1:.4f} ± {std_f1:.4f}")
    logging.info(f"Golden Metric: {avg_f1:.4f}")
    
    logging.info("="*60)
    logging.info(f"\nAll output models and charts are located in: {output_dir}/")
    logging.info(f"Log file saved to: {log_path}")
