import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.sparse as sp_sparse

from collections import defaultdict, Counter


class KPNN(tf.keras.Model):
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
        tf.set_random_seed(config['seed'])
        self.alpha = config['hyperparams']['alpha']
        self.lambda_ = config['hyperparams']['lambda_']
        self.optim_name = config['hyperparams']['optimizer']
        self.tpm_normalization = config['normalization']['TPM']
        self.gene_dropout = config['hyperparams']['gene_dropout']
        self.node_dropout = config['hyperparams']['node_dropout']
        self.num_grad_epsilon = config['hyperparams']['num_grad_epsilon']
        
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

        with tf.name_scope("error"):
            error = tf.abs(y_true - y_hat) * y_weights
            tf.summary.scalar("mean", tf.reduce_mean(error))

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_hat), y_true), tf.float64))
            tf.summary.scalar("mean", tf.reduce_mean(accuracy))

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
            'error': error,
            'accuracy': accuracy
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