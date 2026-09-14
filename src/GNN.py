
import pandas as pd
import torch 
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from Graph_data import GraphData
from sklearn.model_selection import train_test_split

class GNN(nn.Module):
    def __init__(self, input_features , hidden_features ,num_classes,dropout):
        super().__init__()
        self.conv1 = GCNConv(input_features,hidden_features)
        self.conv2 = GCNConv(hidden_features,hidden_features)
        self.conv3 = GCNConv(hidden_features,hidden_features)
        self.conv4 = GCNConv(hidden_features,hidden_features)
        self.conv5 = GCNConv(hidden_features,hidden_features)
      
        self.classifier = nn.Linear(
            hidden_features,
            num_classes)
        self.dropout = dropout

    def forward(self,x,edge_index):

        # First GCN layer
        x = self.conv1(x,edge_index)

        x = F.relu(x)
        x = F.dropout(x,p=self.dropout,training=self.training)

        # Second GCN layer
        x = self.conv2(x,edge_index)

        x = F.relu(x)
        x = F.dropout(x,p=self.dropout,training=self.training)
        # Second GCN layer
        x = self.conv3(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x,p=self.dropout,training=self.training)

        x = self.conv4(x,edge_index)       
        x = F.relu(x)
        x = F.dropout(x,p=self.dropout,training=self.training)

        x = self.conv5(x,edge_index)       
        x = F.relu(x)
        x = F.dropout(x,p=self.dropout,training=self.training)
        

        # Classification layer
        x = self.classifier(x)

        return x

class GNNTrainer():
    def __init__(self,data,learning_rate=0.01,dropout=0.5,epoches = 200):
            self.learning_rate = learning_rate
            self.dropout = dropout
            self.data = data
            self.epoches = epoches
                    

    def create_mask (self):        
            self.data.y = self.data.y.long()

            self.num_node = self.data.x.size(0)
            self.labels = self.data.y.numpy()
            print("size num node : ",self.num_node)
            self.index = torch.arange(self.num_node)
            print("________________________________________________\n lables : ",self.data.y)
            print("Label dtype:", self.data.y.dtype)

            train_indices, test_indices = train_test_split(
                self.index.numpy(),
                test_size=0.2,
                random_state=42,
                stratify=self.labels)

            train_indices = torch.tensor(train_indices)
            test_indices = torch.tensor(test_indices)

            train_mask = torch.zeros(self.num_node,dtype=bool)
            test_mask = torch.zeros(self.num_node,dtype=bool)

            train_mask[train_indices] = True
            test_mask[test_indices] = True

            self.data.train_mask = train_mask
            self.data.test_mask = test_mask

            print("Training nodes:", train_mask.sum().item())
            print("Testing nodes:", test_mask.sum().item())


    def training(self):
            self.create_mask()
            self.input_feature = self.data.x.shape[1]
            
            
            self.model = GNN(
                input_features=self.input_feature,
                hidden_features=32,
                num_classes=2,
                dropout=self.dropout
            )

            print(self.model)
            


            learning_rate = self.learning_rate
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=5e-4
            )

            #  Training 
            self.epoches = 200

            for epoch in range(self.epoches):

                self.model.train()

                optimizer.zero_grad()

                # Forward pass
                output = self.model(
                    self.data.x,
                    self.data.edge_index
                )

                # Only calculate loss on training nodes
                loss = criterion(
                    output[self.data.train_mask],
                    self.data.y[self.data.train_mask]
                )

                # Backpropagation
                loss.backward()

                optimizer.step()

                # Print progress
                if (epoch + 1) % 20 == 0:

                    print(
                        f"Epoch {epoch + 1:03d} | "
                        f"Loss: {loss.item():.4f}"
                    )


            # ============================================================
            # 6. TEST MODEL
            # ============================================================
    def testing(self):
            
            self.model.eval()

            with torch.no_grad():

                output = self.model(
                    self.data.x,
                    self.data.edge_index
                )

                predictions = output.argmax(
                    dim=1
                )

                correct = (
                    predictions[self.data.test_mask]
                    ==
                    self.data.y[self.data.test_mask]
                ).sum()

                total = self.data.test_mask.sum()

                accuracy = correct.float() / total

            print()
            print("==============================")
            print("Test Results")
            print("==============================")
            print("Correct:", correct.item())
            print("Total:", total.item())
            print(
                f"Test Accuracy: {accuracy.item() * 100:.2f}%"
            )
    def run(self):
        print("---------------------------------\n Training started\n")
        self.training()
        print("---------------------------------\n Training compleated\n")
        #save the file
        torch.save(self.model.state_dict(),f='GNN_model.pth')

        print("---------------------------------\n testing started\n")
        self.testing()
        print("---------------------------------\n resting completed\n")

         


# garaph data
Graph_d = GraphData()
data = Graph_d.get_data()