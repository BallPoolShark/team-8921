"""Graph construction module for Graph Neural Network (GNN) analysis.

This module builds a transaction network graph from account transaction data.
The graph represents accounts as nodes and transactions as edges, enabling
Graph Neural Network analysis to detect complex money laundering patterns.

The graph construction process:
    1. Identifies all relevant nodes (candidate accounts and their neighbors)
    2. Filters transactions within the subgraph
    3. Creates edge indices and edge attributes
    4. Computes initial node features (in/out degree)
    5. Applies one-hot encoding for channel types

Attributes:
    graph_data (torch_geometric.data.Data): PyTorch Geometric graph object with
        node features (x), edge indices, and edge attributes.
    acct_to_idx (dict): Mapping from account IDs to node indices in the graph.

Example:
    This module is executed as part of the preprocessing pipeline::

        $ python Preprocess/cell4.py
        開始建圖...
        子圖建置完成: 1500000 edges, 75000 nodes
"""

print("開始建圖...")

def BuildSubGraphSafe(df_txn, candidate_accts):
    """Build a transaction subgraph for candidate accounts.

    This function constructs a PyTorch Geometric Data object representing
    the transaction network. It includes candidate accounts and their
    transaction partners, with edges representing money flows.

    Args:
        df_txn (pd.DataFrame): Transaction DataFrame with columns from_acct,
            to_acct, txn_amt, and channel_type.
        candidate_accts (array-like): List of account IDs to include as
            core nodes in the subgraph.

    Returns:
        tuple: A tuple containing:
            - graph_data (torch_geometric.data.Data): Graph with:
                * x: Node features [out_degree, in_degree] (shape: [N, 2])
                * edge_index: Edge connectivity (shape: [2, E])
                * edge_attr: Edge features [txn_amt, channel_onehot] (shape: [E, F])
            - acct_to_idx (dict): Account ID to node index mapping

    Note:
        The function processes data in batches of 100,000 edges to manage
        memory efficiently. Channel types are one-hot encoded as edge attributes.
        Nodes with no edges will have zero in/out degree features.

    Example:
        >>> graph, idx_map = BuildSubGraphSafe(txn_allowed, alert['acct'].unique())
        >>> print(f"Graph has {graph.num_nodes} nodes and {graph.num_edges} edges")
        Graph has 50000 nodes and 1000000 edges
    """
    cand_set = set(candidate_accts)
    rel_from = set(df_txn[df_txn['to_acct'].isin(cand_set)]['from_acct'])
    rel_to = set(df_txn[df_txn['from_acct'].isin(cand_set)]['to_acct'])
    all_nodes = cand_set.union(rel_from, rel_to)
    
    mask = df_txn['from_acct'].isin(all_nodes) & df_txn['to_acct'].isin(all_nodes)
    df_f = df_txn[mask][['from_acct', 'to_acct', 'txn_amt', 'channel_type']].copy()
    
    acct_to_idx = {acct: idx for idx, acct in enumerate(all_nodes)}
    
    batch_size = 100_000
    edge_index_list = []
    edge_attr_list = []
    
    # channel one-hot
    channel_dummies = pd.get_dummies(df_f['channel_type'], prefix='ch').values.astype(float)
    
    for start in range(0, len(df_f), batch_size):
        batch = df_f.iloc[start:start + batch_size]
        from_idx = batch['from_acct'].map(acct_to_idx).values
        to_idx = batch['to_acct'].map(acct_to_idx).values
        edge_index_list.append(np.stack([from_idx, to_idx]))
        
        amt = batch['txn_amt'].values.reshape(-1, 1)
        ch_batch = channel_dummies[start:start + batch_size]
        edge_attr_list.append(np.hstack([amt, ch_batch]))
    
    edge_index = np.concatenate(edge_index_list, axis=1)
    edge_attr = np.concatenate(edge_attr_list, axis=0)
    
    edge_index = torch.from_numpy(edge_index).long()
    edge_attr = torch.from_numpy(edge_attr).float()
    
    out_deg = torch.bincount(edge_index[0], minlength=len(all_nodes))
    in_deg = torch.bincount(edge_index[1], minlength=len(all_nodes))
    x = torch.stack([out_deg, in_deg], dim=1).float()
    
    del df_f, edge_index_list, edge_attr_list, channel_dummies
    gc.collect()
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), acct_to_idx

candidate_accts = pd.concat([alert['acct'], pred['acct']], ignore_index=True).unique()
graph_data, acct_to_idx = BuildSubGraphSafe(txn_allowed, candidate_accts)
print(f"子圖建置完成: {graph_data.num_edges} edges, {graph_data.num_nodes} nodes")
gc.collect()