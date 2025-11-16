"""Graph Neural Network (GNN) feature extraction module.

This module implements a Graph Convolutional Network (GCN) to learn
node embeddings from the transaction graph. The learned embeddings
capture network structure and are used as additional features for
the downstream XGBoost classifier.

The GNN architecture:
    1. Multi-layer GCN with batch normalization
    2. ReLU activation and dropout for regularization
    3. Unsupervised training using edge reconstruction loss
    4. Extraction of 64-dimensional node embeddings

The embeddings are trained to preserve graph structure by minimizing
the distance between connected nodes' representations.

Attributes:
    gnn_features (pd.DataFrame): DataFrame containing GNN-derived features
        with columns gnn_feat_0 to gnn_feat_63 for each account.

Example:
    This module is executed as part of the model training pipeline::

        $ python Model/cell5.py
        開始 GNN 特徵提取...
        訓練 GNN (CPU, 25 epochs)...
          Epoch 10/25, Loss: 0.0342
          Epoch 20/25, Loss: 0.0198
        GNN 特徵合併完成！最終特徵 shape: (50000, 214)
        Cell 5 完成！
"""

class GNNModel(torch.nn.Module):
    """Graph Convolutional Network model for node embedding learning.

    This class implements a multi-layer GCN using PyTorch Geometric's GCNConv
    layers. The model learns to embed nodes into a lower-dimensional space
    while preserving graph structure.

    Args:
        input_dim (int): Dimension of input node features.
        hidden_dim (int, optional): Dimension of hidden layers. Defaults to 64.
        output_dim (int, optional): Dimension of output embeddings. Defaults to 64.
        num_layers (int, optional): Number of GCN layers. Defaults to 3.

    Attributes:
        convs (torch.nn.ModuleList): List of GCN convolutional layers.
        bns (torch.nn.ModuleList): List of batch normalization layers.

    Example:
        >>> model = GNNModel(input_dim=2, hidden_dim=64, output_dim=64)
        >>> embeddings = model(node_features, edge_index)
        >>> print(embeddings.shape)
        torch.Size([num_nodes, 64])
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=64, num_layers=3):
        """Initialize the GNN model with specified architecture.

        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int, optional): Dimension of hidden layers. Defaults to 64.
            output_dim (int, optional): Dimension of output embeddings. Defaults to 64.
            num_layers (int, optional): Number of GCN layers. Defaults to 3.
        """
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # 不傳任何 edge 參數，只用 edge_index
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.bns.append(torch.nn.BatchNorm1d(output_dim))

    def forward(self, x, edge_index):
        """Perform forward pass through the GCN layers.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, input_dim].
            edge_index (torch.Tensor): Edge connectivity of shape [2, num_edges].

        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, output_dim].

        Note:
            Applies GCN convolution, batch normalization, ReLU activation,
            and dropout (p=0.3) between layers. No activation after final layer.
        """
        # 只傳 edge_index，不傳 edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
        return x

def ExtractGNNFeatures(graph_data, acct_to_idx, target_accts, epochs=25):
    """Extract GNN embeddings for target accounts.

    This function trains a GNN model on the transaction graph using
    unsupervised learning (edge reconstruction), then extracts node
    embeddings for the specified target accounts.

    Args:
        graph_data (torch_geometric.data.Data): Graph object with node
            features (x) and edge indices.
        acct_to_idx (dict): Mapping from account IDs to node indices.
        target_accts (array-like): List of account IDs to extract
            embeddings for.
        epochs (int, optional): Number of training epochs. Defaults to 25.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - acct: Account ID
            - gnn_feat_0 to gnn_feat_{dim-1}: GNN embedding features

    Note:
        Training uses MSE loss between embeddings of connected nodes.
        Accounts not in the graph receive zero-vector embeddings.
        The model is trained on CPU with Adam optimizer (lr=0.01).

    Example:
        >>> gnn_df = ExtractGNNFeatures(graph, acct_map, target_accounts, epochs=25)
        訓練 GNN (CPU, 25 epochs)...
          Epoch 10/25, Loss: 0.0342
        >>> gnn_df.shape
        (10000, 65)  # 64 features + 1 acct column
    """
    device = 'cpu'
    model = GNNModel(graph_data.x.shape[1]).to(device)
    graph_data = graph_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print(f"訓練 GNN (CPU, {epochs} epochs)...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(graph_data.x, graph_data.edge_index)  # 不傳 edge_attr
        src = embeddings[graph_data.edge_index[0]]
        dst = embeddings[graph_data.edge_index[1]]
        loss = F.mse_loss(src, dst)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        embeddings = model(graph_data.x, graph_data.edge_index).cpu().numpy()  # 不傳 edge_attr
    
    gnn_dict = {}
    zero_vec = np.zeros(embeddings.shape[1])
    for acct in target_accts:
        idx = acct_to_idx.get(acct)
        gnn_dict[acct] = embeddings[idx] if idx is not None else zero_vec
    
    cols = [f'gnn_feat_{i}' for i in range(embeddings.shape[1])]
    gnn_df = pd.DataFrame.from_dict(gnn_dict, orient='index', columns=cols)
    gnn_df.reset_index(inplace=True)
    gnn_df.rename(columns={'index': 'acct'}, inplace=True)
    
    return gnn_df

# 執行
all_target_accts = pd.concat([feat['acct'], pred['acct']], ignore_index=True).unique()
print("開始 GNN 特徵提取...")
gnn_features = ExtractGNNFeatures(graph_data, acct_to_idx, all_target_accts, epochs=25)

feat = feat.merge(gnn_features, on='acct', how='left').fillna(0)
print(f"GNN 特徵合併完成！最終特徵 shape: {feat.shape}")
del graph_data, gnn_features
torch.cuda.empty_cache()
gc.collect()
print("Cell 5 完成！可跑 Cell 6")