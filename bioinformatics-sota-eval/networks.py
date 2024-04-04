import copy
import torch
import pykegg
import numpy as np
import pandas as pd
import requests_cache
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F
import scipy.sparse as sp_sparse
import tensorflow.keras as keras

from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from collections import defaultdict, Counter

torch.manual_seed(42)


class ANN(nn.Module):
    def __init__(self, genes: int, config: dict):
        super(ANN, self).__init__()
        
        name = self.__class__.__name__.lower()
        
        num_input = len(genes)
        self.fc1 = torch.nn.Linear(num_input, 4096)
        self.fc2 = torch.nn.Linear(4096, 1024)
        self.fc3 = torch.nn.Linear(1024, 256)
        self.fc4 = torch.nn.Linear(256, 64)
        self.fc5 = torch.nn.Linear(64, 16)
        self.fc6 = torch.nn.Linear(16, 1)
        
        self.loss = nn.BCEWithLogitsLoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=config[name]['alpha'])

    def forward(self, x):
        layers = list(self.children())[:-2]
        
        for layer in layers:
            x = F.relu(layer(x))
            
        return self.fc6(x)
    
    
class GNN(nn.Module):
    def __init__(self, config: dict):
        super(GNN, self).__init__()
        
        name = self.__class__.__name__.lower()
        graph = self._build_graph(config[name]['pathway_id'])
        
        num_node_features = graph.x.size()
        
        self.conv1 = GCNConv(num_node_features, 256)
        self.conv2 = GCNConv(256, 256)
        self.fc = nn.Linear(256, 1)
        
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.loss = nn.MSELoss()
        self.dropout_rate = config[name]['dropout_rate']
        self.optim = torch.optim.Adam(self.parameters(), lr=config[name]['alpha'])
    
    # TODO: Finish building the graph
    # Graph should contain the adjacency matrix and node features
    def _build_graph(self, pathway_id: str) -> Data:
        requests_cache.install_cache('pykegg_cache')
        graph = pykegg.KGML_graph(pid=pathway_id)
        
        node_df = graph.get_nodes()
        edge_df = graph.get_edges()
        
        node_features = node_df[["graphics_name", "type"]]

        node_type_mapping = {node_type: idx for idx, node_type in enumerate(node_features["type"].unique())}

        # Convert node types to integer indices
        node_features["type"] = node_features["type"].map(node_type_mapping)

        # Create a dictionary to map node IDs to integer indices
        node_id_mapping = {node_id: idx for idx, node_id in enumerate(node_df["id"])}

        # Create edge index tensor
        edge_index = torch.tensor(edge_df[["source", "target"]].applymap(node_id_mapping).values.T, dtype=torch.long)

        # Create node feature tensor
        node_feature_tensor = torch.tensor(node_features["type"].values, dtype=torch.long)

        # Create PyTorch Geometric Data object
        data = Data(x=node_feature_tensor, edge_index=edge_index)
    
    def forward(self, data: Data) -> torch.tensor:
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate)
                
        return self.fc(x)


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, relation_file, bias=True) -> None:
        super(MaskedLinear, self).__init__(in_features, out_features, bias)

        mask = self.read_relation(relation_file)
        self.register_buffer('mask', mask)

        self.iter = 0

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

    def read_relation(self, relation_file):
        mask = []
        
        with open(relation_file, 'r') as f:
            for line in f:
                l = [int(x) for x in line.strip().split(',')]
                for item in l:
                    assert item == 1 or item == 0
                mask.append(l)
                
        return Variable(torch.Tensor(mask))
    
class PGNN(nn.Module):
    def __init__(self, config: dict) -> None:
        super(PGNN, self).__init__()
        
        name = self.__class__.__name__.lower()
        
        num_cellline_genes = config[name]['num_cellline_genes']
        num_drug_genes = config[name]['num_drug_genes']
        num_pathways = config[name]['num_pathways']
        
        self.cell_drug_masked_fc = MaskedLinear(num_cellline_genes + num_drug_genes, num_pathways, 'data/relation.csv')
        self.fc1 = nn.Linear(num_pathways, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.loss = nn.MSELoss()
        
        sgd_alpha = config[name]['sgd_alpha']
        adam_alpha = config[name]['adam_alpha']
        momentum = config[name]['momentum']
        weight_decay = config[name]['weight_decay']
        dropout_rate = config[name]['dropout_rate']
        
        self.dropout_rate = dropout_rate
        self.optim_adam = torch.optim.Adam(self.parameters(), lr=adam_alpha, weight_decay=weight_decay)
        self.optim_sdg = torch.optim.SGD(self.parameters(), lr=sgd_alpha, momentum=momentum, weight_decay=weight_decay)
        
    def forward(self, cells: torch.tensor, drug: torch.tensor, genes: torch.tensor) -> torch.tensor:
        input_tensor = torch.cat((cells, drug, genes), dim=1)
        
        pathways = F.relu(self.cell_drug_masked_fc(input_tensor))
        
        x = F.dropout(self.fc1(pathways), p=self.dropout_rate)
        x = F.relu(x)
        x = F.dropout(self.fc2(x), p=self.dropout_rate)
        x = F.relu(x)

        return self.fc3(x)
    

class KPNN(keras.Model):
    def __init__(self, edges: pd.DataFrame, genes: list, config: dict) -> None:
        """
        Initialize the KPNN (Knowledge-Primed Neural Network) model.
        
        Args:
            edges (pd.DataFrame): A DataFrame containing the edges (parent-child relationships) of the gene hierarchy.
            genes (list): A list of gene names.
            config (dict): A dictionary containing the configuration parameters for the model.
        """
        
        super(KPNN, self).__init__()
        self._init_hyperparams(config)
        
        self.optimizer = None
        self.ranked_nodes = []
        self.node_weights = {}
        self.node_to_genes = {}
        self.node_to_nodes = {}
        self.total_weight_count = None
        self.network_edges = defaultdict(list)
                        
        self._rank_nodes_in_hierarchy(edges, genes)
        self._create_mappings(edges, genes)
        self._validate_mappings(genes)
        
    def _init_hyperparams(self, config: dict) -> None:
        """
        Initialize the hyperparameters from the configuration dictionary.

        Args:
            config (dict): The configuration dictionary.
        """
        name = self.__class__.__name__.lower()
        
        tf.set_random_seed(config['seed'])
        self.alpha = config[name]['alpha']
        self.tpm_normalization = config['TPM']
        self.lambda_ = config[name]['lambda_']
        self.minmax_normalization = config['minmax']
        self.optim_name = config[name]['optimizer']
        self.gene_dropout = config[name]['gene_dropout']
        self.node_dropout = config[name]['node_dropout']
        self.num_grad_epsilon = config[name]['num_grad_epsilon']
        
    def _rank_nodes_in_hierarchy(self, edges: pd.DataFrame, genes: list) -> list:
        """
        Rank the nodes in the gene hierarchy based on their position, starting from leaf nodes to the root node.
        
        Args:
            edges (pd.DataFrame): A DataFrame containing the edges (parent-child relationships) of the gene hierarchy.
            genes (list): A list of gene names.
            
        Returns:
            list: A list of ranked nodes in the hierarchy.
        """
        
        remaining_edges = copy.deepcopy(edges)
        prev_remaining_edges_count = remaining_edges.shape[0] + 1
        
        iteration = 1
        while remaining_edges.shape[0] < prev_remaining_edges_count:
            prev_remaining_edges_count = remaining_edges.shape[0]
            children_nodes = set(remaining_edges['child'].tolist())
            parent_nodes = set(remaining_edges['parent'].tolist())
            leaf_nodes = sorted(children_nodes - parent_nodes)
            
            if iteration > 1:
                self.ranked_nodes.extend(leaf_nodes)
            else:
                assert set(leaf_nodes).issubset(set(genes)), "Some genes not found in the data"
                
            remaining_edges = remaining_edges.loc[~remaining_edges['child'].isin(leaf_nodes)]
            iteration += 1
        
        parent_nodes = set(edges['parent'].tolist())
        children_nodes = set(edges['child'].tolist())
        self.ranked_nodes.extend(parent_nodes - children_nodes)
            
    def _create_mappings(self, edges: pd.DataFrame, genes: list) -> tuple:
        """
        Create mappings between nodes, genes, and weight indices.
        
        Args:
            edges (pd.DataFrame): A DataFrame containing the edges (parent-child relationships) of the gene hierarchy.
            genes (list): A list of gene names.
        """
        
        for _, row in edges.iterrows():
            parent = row['parent']
            child = row['child']
            self.network_edges[parent].append(child)

        weight_counter = 0
        for node in self.ranked_nodes:
            self.node_to_genes[node] = []
            self.node_to_nodes[node] = []
            self.node_weights[node] = []
            
            for edge in self.network_edges[node]:
                self.node_weights[node].append(weight_counter)
                weight_counter += 1
                
                if edge in genes:
                    self.node_to_genes[node].append(genes.index(edge))
                elif edge in self.ranked_nodes:
                    self.node_to_nodes[node].append(self.ranked_nodes.index(edge))
                else:
                    raise ValueError(f"Node {edge} not found in the data")
                                
    def _validate_mappings(self, genes: list) -> None:
        """
        Validate the created mappings between nodes, genes, and weight indices.
        
        Args:
            genes (list): A list of gene names.
        """
        
        self.total_weight_count = sum(len(self.node_weights[node]) for node in self.ranked_nodes)        
        assert max(max(weight_list) for weight_list in self.node_weights.values()) == self.total_weight_count - 1
        
        for node in self.ranked_nodes:
            total_associations = len(self.node_to_genes[node]) + len(self.node_to_nodes[node])
            
            assert total_associations == len(self.node_weights[node])            
            assert total_associations == len(self.network_edges[node])
            
            node_neighbors = set(self.network_edges[node]) & set(self.ranked_nodes[i] for i in self.node_to_nodes[node])
            assert len(self.node_to_nodes[node]) == len(node_neighbors)
            
            gene_neighbors = set(self.network_edges[node]) & set(genes[i] for i in self.node_to_genes[node])
            assert len(self.node_to_genes[node]) == len(gene_neighbors)

    def setup_network(self, x_train: sp_sparse.csc_matrix, edges: pd.DataFrame, outputs: list) -> dict:
        """
        Set up the neural network architecture and define placeholders, variables, and operations.
        
        Args:
            x_train (sp_sparse.csc_matrix): A sparse matrix containing the input gene expression data.
            edges (pd.DataFrame): A DataFrame containing the edges (parent-child relationships) of the gene hierarchy.
            outputs (list): A list of output node names.
            
        Returns:
            dict: A dictionary containing the necessary tensors and operations for training and evaluation.
        """
        
        # Define placeholders for dropout rates
        node_dropout = tf.cast(tf.placeholder_with_default(input=1.0, shape=[], name="node_dropout"), tf.float64)
        gene_dropout = tf.cast(tf.placeholder_with_default(input=1.0, shape=[], name="gene_dropout"), tf.float64)

        # Define placeholders for true labels and weights
        y_true = tf.placeholder(name="y_true", shape=[len(outputs), None], dtype=tf.float64)
        y_weights = tf.placeholder(name="y_weights", shape=[len(outputs), None], dtype=tf.float64)

        with tf.name_scope("input_genes"):
            genes_orig = tf.placeholder(name="genes", shape=[x_train.shape[0], None], dtype=tf.float64)
            genes = tf.cond(
                tf.greater(gene_dropout, 0),
                lambda: tf.nn.dropout(x=genes_orig, rate=gene_dropout),
                lambda: genes_orig
            )
            
        # Define placeholders for numerical approximation
        num_approx_plus = tf.placeholder_with_default(input="", shape=[], name="num_approx_up")
        num_approx_minus = tf.placeholder_with_default(input="", shape=[], name="num_approx_down")
        num_approx_vec = tf.reshape(tf.cast(tf.tile([self.num_grad_epsilon], [tf.shape(genes_orig)[1]],), tf.float64), [1, tf.shape(genes_orig)[1]])

        with tf.name_scope("weights"):
            weights = tf.Variable(tf.random_normal([self.total_weight_count, 1], dtype=tf.float64, name="init_weights"), name="weights", dtype=tf.float64)
            tf.summary.scalar("mean", tf.reduce_mean(weights))
            tf.summary.histogram("histogram", weights)

        with tf.name_scope("intercept"):
            intercept_weights = tf.Variable(tf.random_normal([len(self.ranked_nodes)], dtype=tf.float64, name="init_intercepts"), name="intercepts", dtype=tf.float64)
            tf.summary.scalar("mean", tf.reduce_mean(intercept_weights))
            tf.summary.histogram("histogram", intercept_weights)

        node_values = {}
        genes_unstacked = tf.unstack(genes)

        # Iterate over ranked nodes and compute node values
        for node in self.ranked_nodes:
            weights_x = tf.slice(weights, [self.node_weights[node][0], 0], [len(self.node_weights[node]), 1])

            gene_features = [genes_unstacked[x] for x in self.node_to_genes[node]]
            node_features = [node_values[self.ranked_nodes[nidx]] for nidx in self.node_to_nodes[node]]
            features = tf.stack(gene_features + node_features)

            weighted_sum = tf.matmul(tf.transpose(weights_x), features)
            weighted_sum += tf.slice(intercept_weights, [self.ranked_nodes.index(node)], [1])

            node_values[node] = tf.nn.sigmoid(weighted_sum) if node not in outputs else weighted_sum

            # Apply numerical approximation
            node_values[node] = tf.cond(tf.equal(num_approx_plus, tf.constant(node)), lambda: node_values[node] + num_approx_vec, lambda: node_values[node])
            node_values[node] = tf.cond(tf.equal(num_approx_minus, tf.constant(node)), lambda: node_values[node] - num_approx_vec, lambda: node_values[node])
            node_values[node] = tf.unstack(node_values[node])[0]

            # Apply dropout to non-output nodes
            if self.node_dropout > 0 and node not in outputs:
                print(f'Performing dropout on {node}')
                parents = edges[edges['child'].str.match("^" + node + "$")]['parent'].tolist()
                children = len(set(edges[edges['parent'].isin(parents)]["child"].tolist()))

                if children == 1:
                    print(f'{node}\'s parents have 1 child - dropout skipped')
                elif children == 2 and self.node_dropout > 0:
                    print(f'{node}\'s parents have 2 children - dropout adjusted to 0.1 or {self.node_dropout}')
                    node_values[node] = tf.nn.dropout(x=node_values[node], rate=tf.maximum(node_dropout, 0.1))
                elif children == 3 and self.node_dropout > 0:
                    print(f'{node}\'s parents have 3 children - dropout adjusted to 0.3 or {self.node_dropout}')
                    node_values[node] = tf.nn.dropout(x=node_values[node], rate=tf.maximum(node_dropout, 0.3))
                else:
                    node_values[node] = tf.nn.dropout(x=node_values[node], rate=node_dropout)

        with tf.name_scope("regularization"):
            regularization = (self.lambda_ * tf.nn.l2_loss(weights)) / tf.cast(tf.shape(y_true)[1], tf.float64)
            tf.summary.scalar("value", regularization)

        with tf.name_scope("xentropy"):
            xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.stack([node_values[x] for x in outputs]), labels=y_true) * y_weights
            tf.summary.scalar("mean", tf.reduce_mean(xentropy))
            tf.summary.histogram("histogram", xentropy)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(xentropy) + regularization
            tf.summary.scalar("value", loss)

        with tf.name_scope("y_hat"):
            y_hat = tf.nn.sigmoid(tf.stack([node_values[x] for x in outputs]))


        with tf.name_scope("optimizer"):
            optim_class = getattr(tf.train, self.optim_name)
            optimizer = optim_class(learning_rate=self.alpha)
            trainer = optimizer.minimize(loss)

        return {
            'trainer': trainer, 
            'loss': loss,
            'genes_orig': genes_orig,
            'y_true': y_true,
            'y_weights': y_weights,
            'y_hat': y_hat,
            'node_dropout': node_dropout,
            'gene_dropout': gene_dropout,
        }
            
    def get_weight_matrix(self, input_labels: np.array) -> np.array:
        """
        Compute the weight matrix for the input labels based on their frequency.

        Args:
            input_labels (np.array): A numpy array containing the input labels.

        Returns:
            np.array: A weight matrix for the input labels.
        """
        
        label_combinations = ["".join(map(str, x)) for x in input_labels.astype(int).T]
        
        combination_counts = Counter(label_combinations)        
        total_combinations = len(label_combinations)
        
        combination_weights = {combo: (1.0 / (count / total_combinations)) / len(combination_counts)
            for combo, count in combination_counts.items()
        }
        
        weight_matrix = np.array([[combination_weights[combo] for combo in label_combinations]] * input_labels.shape[0])
        
        return weight_matrix