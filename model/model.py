"""
Model Definition Module
-----------------------
本模組定義了用於詐欺偵測的圖神經網路模型架構 (FraudGNN)。

架構特色：
結合了 GraphSAGE 的聚合能力與 GAT (Graph Attention Network) 的注意力機制，
旨在從交易網路中捕捉異常的結構模式。

Usage:
    from Model.model import FraudGNN
"""

import torch.nn.functional as F
import torch
from torch_geometric.nn import SAGEConv, GATConv

class FraudGNN(torch.nn.Module):
    """
    詐欺偵測圖神經網路模型 (Graph Neural Network for Fraud Detection)。

    架構設計：
    採用 GraphSAGE 與 GAT 的混合架構：
    1. **GraphSAGE Layers**: 用於高效聚合鄰居節點資訊，捕捉局部結構特徵，適合處理大規模圖數據。
    2. **GAT Layer**: 在深層引入注意力機制 (Attention)，賦予不同鄰居不同的權重，強化關鍵異常節點的影響力。
    3. **Classification Head**: 透過全連接層 (MLP) 將圖特徵映射為二元分類結果。

    Attributes:
        convs (nn.ModuleList): 儲存 GraphSAGE 卷積層。
        bns (nn.ModuleList): 儲存 Batch Normalization 層。
        gat (GATConv): GAT 注意力卷積層。
        fc1, fc2 (nn.Linear): 全連接分類層。
    """
    def __init__(self, in_channels, hidden_channels=128, num_layers=3, dropout=0.3):
        """
        初始化模型架構。

        Args:
            in_channels (int): 輸入特徵的維度 (即 feature.csv 的欄位數)。
            hidden_channels (int): 隱藏層的維度 (Hidden Dimension)。預設為 128。
            num_layers (int): GraphSAGE 層的數量 (不含 GAT 層)。預設為 3。
            dropout (float): Dropout 機率，用於防止過擬合。預設為 0.3。
        """
        super(FraudGNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # 第一層
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # 中間層
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # 最後一層使用GAT（注意力機制）
        self.gat = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.bn_gat = torch.nn.BatchNorm1d(hidden_channels)
        
        # 分類層
        self.fc1 = torch.nn.Linear(hidden_channels, 64)
        self.fc2 = torch.nn.Linear(64, 2)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """
        模型前向傳播 (Forward Pass)。

        Args:
            x (torch.Tensor): 節點特徵矩陣。形狀為 [num_nodes, in_channels]。
            edge_index (torch.LongTensor): 圖的邊索引 (COO format)。形狀為 [2, num_edges]。

        Returns:
            torch.Tensor: 模型輸出的 Logits (未經過 Softmax)。形狀為 [num_nodes, 2]。
        """
        # GraphSAGE層
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT層（注意力機制捕捉重要鄰居）
        x = self.gat(x, edge_index)
        x = self.bn_gat(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 分類層
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x