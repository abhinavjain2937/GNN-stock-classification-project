import pandas as pd
import torch 
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from Graph_data import GraphData
from sklearn.model_selection import train_test_split
from GNN import GNNTrainer

# garaph data
Graph_d = GraphData()
data = Graph_d.get_data()

# GNN training class loaading
Train_gnn = GNNTrainer(data=data,learning_rate=0.01)
result = Train_gnn.run()


