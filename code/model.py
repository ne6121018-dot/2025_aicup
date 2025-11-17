import torch.nn.functional as F
import torch
from torch_geometric.nn import SAGEConv, GATConv

class FraudGNN(torch.nn.Module):
    """
    詐欺偵測GNN模型
    使用GraphSAGE + GAT的混合架構
    """
    def __init__(self, in_channels, hidden_channels=128, num_layers=3, dropout=0.3):
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