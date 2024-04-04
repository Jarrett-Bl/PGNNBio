import os
import torch
import tables
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class GeneBCMatrix:
    """
    A class representing a gene-barcode matrix.
    """

    def __init__(self, gene_ids: np.ndarray, gene_names: np.ndarray, barcodes: np.ndarray, matrix: sp_sparse.csc_matrix) -> None:
        """
        Initialize a GeneBCMatrix instance.

        Args:
            gene_ids (np.ndarray): A numpy array of gene IDs.
            gene_names (np.ndarray): A numpy array of gene names.
            barcodes (np.ndarray): A numpy array of barcodes.
            matrix (sp_sparse.csc_matrix): A sparse matrix representing the gene-barcode matrix.
        """
        self.gene_ids = gene_ids
        self.gene_names = gene_names
        self.barcodes = barcodes
        self.matrix = matrix
    

class DataPipeline:    
    @staticmethod
    def load_input_data(input_file: str, config: dict) -> tuple:
        """
        Load the input gene expression data from an HDF5 file.

        Args:
            input_file (str): The path to the HDF5 file containing the gene expression data.
            config (dict): A dictionary containing the normalization configuration.

        Returns:
            tuple: A tuple containing the normalized gene expression data, gene names, and barcode names.
        """
        
        h5f = tables.open_file(input_file, 'r')
        genome = h5f.list_nodes(h5f.root)[0]._v_name
        h5f.close()
        
        gene_bc_matrix = DataPipeline.get_gene_bc_matrix(input_file, genome)
        
        barcodes_x = gene_bc_matrix.barcodes.tolist()
        
        genes_x = gene_bc_matrix.gene_names.tolist()
        genes_x = [x + "_gene" for x in genes_x]
        
        data = gene_bc_matrix.matrix.astype('float64')
        
        tpm_normalization = config['TPM']
        minmax_normalization = config['minmax']
        
        if tpm_normalization:
            data = DataPipeline._normalize_TPM(data)
        if minmax_normalization:
            data = DataPipeline._normalize_minmax(data)
        
        return data, genes_x, barcodes_x
    
    @staticmethod
    def get_gene_bc_matrix(input_file: str, genome: str) -> GeneBCMatrix:
        """
        Retrieve the gene-barcode matrix from the input HDF5 file.

        Args:
            input_file (str): The path to the HDF5 file containing the gene expression data.
            genome (str): The name of the genome group in the HDF5 file.

        Returns:
            gene_bc_matrix: A named tuple containing the gene IDs, gene names, barcodes, and sparse matrix.
        """
        
        with tables.open_file(input_file, 'r') as f:
            try:
                group = f.get_node(f.root, genome)
                gene_ids = getattr(group, 'genes').read().astype(str)
                gene_names = getattr(group, 'gene_names').read().astype(str)
                barcodes = getattr(group, 'barcodes').read().astype(str)
                data = getattr(group, 'data').read()
                indices = getattr(group, 'indices').read()
                indptr = getattr(group, 'indptr').read()
                shape = getattr(group, 'shape').read()
                matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
                return GeneBCMatrix(gene_ids, gene_names, barcodes, matrix)
            except tables.NoSuchNodeError:
                raise ValueError(f"Genome {genome} not found in {input_file}")
    
    @staticmethod
    def _normalize_TPM(data: sp_sparse.csc_matrix) -> sp_sparse.csc_matrix:
        """
        Normalize the gene expression data using Transcripts Per Million (TPM) and log1p transformation.

        Args:
            data (sp_sparse.csc_matrix): The sparse gene expression matrix.

        Returns:
            sp_sparse.csc_matrix: The normalized gene expression matrix.
        """
        
        for col_idx in range(data.shape[1]):
            non_zero_values = data.data[data.indptr[col_idx] : data.indptr[col_idx+1]]
            normalized_values = non_zero_values / non_zero_values.sum() * 1e6
            data.data[data.indptr[col_idx] : data.indptr[col_idx+1]] = normalized_values
        data = data.log1p()
        
        return data
    
    @staticmethod
    def _normalize_minmax(data: sp_sparse.csc_matrix) -> sp_sparse.csc_matrix:
        """
        Normalize the gene expression data using min-max scaling [0-1].

        Args:
            data (sp_sparse.csc_matrix): The sparse gene expression matrix.

        Returns:
            sp_sparse.csc_matrix: The normalized gene expression matrix.
        """
        
        data = data.tocsr()
        scaler = MinMaxScaler()
        
        for i in range(data.shape[0]):
            data[i, :] = scaler.fit_transform(data[i, :].toarray())    
                
        return data
            
    @staticmethod
    def load_data_labels(labels_file: str, barcodes_x: list, output_path: str) -> pd.DataFrame:
        """
        Load the data labels from a CSV file and match them with the input barcodes.

        Args:
            labels_file (str): The path to the CSV file containing the data labels.
            barcodes_x (list): A list of barcodes from the input data.
            output_path (str): The path to the output directory.

        Returns:
            pd.DataFrame: A DataFrame containing the matched data labels.
        """
        
        labels = pd.read_csv(labels_file, sep=',')
        
        barcodes_y = labels['barcode'].tolist()
        # Get the barcodes that are in both the input data and the labels
        barcodes = sorted(set(barcodes_y) & set(barcodes_x))
        
        assert len(barcodes) == len(set(barcodes)), "BARCODES ARE NOT UNIQUE!"
        print("Number of Barcodes found in Y but not X: " + str(len(set(barcodes_y) - set(barcodes_x))))
        print("Number of Barcodes used: " + str(len(barcodes)) + '\n')
        
        # Sort labels based on order of barcodes in input data
        labels = labels.loc[DataPipeline.index_in_list(barcodes, barcodes_y)]
        
        with open(os.path.join(output_path, 'barcodes.txt'), 'w') as f:
            f.write('\n'.join(barcodes))
                    
        return labels, barcodes
    
    @staticmethod     
    def index_in_list(barcodes: list, barcodes_y: list) -> list:
        """
        Retrieve the indices of barcodes from one list based on their order in another list.

        Args:
            barcodes (list): The list of barcodes to retrieve indices for.
            barcodes_y (list): The reference list of barcodes.

        Returns:
            list: A list of indices corresponding to the barcodes in the reference list.
        """
        
        indices, sorted_indices = {}, []
        
        for idx, gene_bc in enumerate(barcodes_y):
            indices[gene_bc] = idx
        
        for idx, gene_bc in enumerate(barcodes):
            sorted_indices.append(indices[gene_bc])
        
        return sorted_indices
    
    @staticmethod
    def load_edge_data(edges_file: str, genes_x: list, output_path: str) -> pd.DataFrame:
        """
        Load the edge data from a CSV file and match it with the input genes.

        Args:
            edges_file (str): The path to the CSV file containing the edge data.
            genes_x (list): A list of genes from the input data.
            output_path (str): The path to the output directory.

        Returns:
            pd.DataFrame: A DataFrame containing the matched edge data.
            list: A list of matched genes.
            list: The original list of genes from the input data.
        """
        
        edges = pd.read_csv(edges_file, sep=',')
        
        parent = edges['parent'].tolist()
        child = edges['child'].tolist()
        # Leaf nodes do not regulate any genes but they themselves are regulated by other genes
        leaf_nodes = list(set(child) - set(parent))
        
        genes = sorted(set(genes_x) & set(leaf_nodes))
        print("Number of Genes used: " + str(len(genes)) + '\n')
        assert len(genes) == len(set(genes)), "genesList non unique!"
        edges = edges[edges['child'].isin(genes) | edges['child'].isin(edges['parent'])]
        
        with open(os.path.join(output_path, 'genes.txt'), 'w') as f:
            f.write('\n'.join(genes))
        
        return edges, genes, genes_x
    
    @staticmethod
    def validate_data(genes: list, edges: pd.DataFrame, barcodes: list, barcodes_x: list, data: sp_sparse.csc_matrix, labels: pd.DataFrame, output_dir: str) -> set:
        """
        Validate the input data, labels, and edges, and ensure that the outputs are present in both the edge data and labels.

        Args:
            genes (list): A list of genes.
            edges (pd.DataFrame): A DataFrame containing the edge data.
            barcodes (list): A list of barcodes.
            barcodes_x (list): A list of barcodes from the input data.
            data (sp_sparse.csc_matrix): The sparse gene expression matrix.
            labels (pd.DataFrame): A DataFrame containing the data labels.
            output_dir (str): The path to the output directory.

        Returns:
            set: A set of output nodes that are present in both the edge data and labels.
        """
        
        labels_lst = labels.columns.tolist()
        parents = edges['parent'].tolist()
        children = edges['child'].tolist()
        
        outputs_y = sorted(set(labels_lst) - set(["barcode"]))
        outputs_edges = sorted(set(parents) - set(children))
        outputs = sorted(set(outputs_y) & set(outputs_edges))
        
        print("Number of Outputs: " + str(len(outputs))  + " --> " + ",".join(outputs))
        print("Class labels: " + ",".join(outputs_y))
        print("Network outputs: " + ",".join(outputs_edges))
        
        assert len(outputs) > 0, "No outputs fitting between y table and edgelist"
                
        with open(os.path.join(output_dir, "outputs.txt"), "w") as f:
            f.write("\n".join(outputs))
        
        outputs_edges_missed = list(set(outputs_edges) - set(outputs))
        input_edges_missed = list(set(set(children) - set(parents)) - set(genes))
        
        while len(outputs_edges_missed) > 0 or len(input_edges_missed) > 0:
            edges = edges[~edges['parent'].isin(outputs_edges_missed)]
            edges = edges[~edges['child'].isin(input_edges_missed)]
            # These are the parents that are not children and also not in outputs (leaves with only out edges - which can only be outputs)
            outputs_edges_missed = list(set(set(parents) - set(children)) - set(outputs))
            # These are the children that are not parents and also not in the gene list (leaves with only in edges - which can only be genes)
            input_edges_missed = list(set(set(children) - set(parents)) - set(genes))

        for x in outputs:
            assert x in parents, x + " missing from edgelist"
            
        edges.to_csv(os.path.join(output_dir, "edges.tsv"), index=False)
        
        # Extract the relevant data
        data = data[:, DataPipeline.index_in_list(barcodes, barcodes_x)]
            
        return outputs, data, labels
    
    @staticmethod
    def generate_datasets(data: sp_sparse.csc_matrix, genes: list, genes_x: list, barcodes: list, labels: pd.DataFrame, config: dict) -> tuple:
        """
        Generate train, validation, and test datasets from the input data, genes, barcodes, and labels.

        Args:
            data (sp_sparse.csc_matrix): The sparse gene expression matrix.
            genes (list): A list of genes.
            genes_x (list): The original list of genes from the input data.
            barcodes (list): A list of barcodes.
            labels (pd.DataFrame): A DataFrame containing the data labels.
            config (dict): A dictionary containing the hyperparameters.

        Returns:
            tuple: A tuple containing the train, validation, and test datasets.
        """
    
        assert labels['barcode'].tolist() == barcodes, "Barcodes in labels do not match barcodes in data"
        
        # Extract relevant genes
        data = data[DataPipeline.index_in_list(genes, genes_x), :]
                
        test_size = config['test_size']
        val_size = config['val_size']

        indices = list(range(len(barcodes)))

        train_indices, tmp_indices = train_test_split(indices, test_size=(test_size + val_size))
        val_indices, test_indices = train_test_split(tmp_indices, test_size=test_size / (test_size + val_size))

        train_x = data[:, train_indices]
        train_Y = labels.iloc[train_indices]['TCR'].to_numpy()
        train_Y = np.expand_dims(train_Y, axis=0)
        
        val_x = data[:, val_indices]
        val_Y = labels.iloc[val_indices]['TCR'].to_numpy()
        val_Y = np.expand_dims(val_Y, axis=0)
        
        test_x = data[:, test_indices]
        test_Y = labels.iloc[test_indices]['TCR'].to_numpy()
        test_Y = np.expand_dims(test_Y, axis=0)
        
        return (train_x, train_Y), (val_x, val_Y), (test_x, test_Y)
    

# Memory efficient dataset that loads data on the fly
class CustomDataset(Dataset):
    def __init__(self, data: sp_sparse.csc_matrix, labels: np.ndarray) -> None:
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return self.data.shape[1]

    def __getitem__(self, index: int) -> tuple:
        x = torch.from_numpy(self.data[:, index].toarray().T).squeeze().float()
        y = torch.from_numpy(self.labels[:, index]).squeeze().float()
        return x, y