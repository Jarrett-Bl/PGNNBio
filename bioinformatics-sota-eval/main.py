import os
import yaml
import torch
import wandb
import numpy as np
import tensorflow as tf

from arguments import get_arguments
from networks import ANN, GNN, PGNN, KPNN
from sklearn.metrics import roc_auc_score
from utils.data_pipeline import DataPipeline
from torch.utils.data import DataLoader, TensorDataset


class Driver:
    def __init__(self, input_file: str, edges_file: str, labels_file: str, output_dir: str) -> None:
        """
        Initialize the Driver class.

        Args:
            input_file (str): The path to the input data file.
            edges_file (str): The path to the edge data file.
            labels_file (str): The path to the labels file.
            output_dir (str): The path to the output directory.
        """
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.model_dir = f'{output_dir}/model'
    
        config_path = 'bioinformatics-sota-eval/utils/config.yaml'
        self.config = self._load_config(config_path)
                
        edges, genes, outputs = self._gather_data(input_file, edges_file, labels_file, self.config)
        
        self.ann = ANN(genes, self.config)
        self.gnn = GNN(self.config)
        self.pgnn = PGNN(self.config)
        self.kpnn = KPNN(edges, genes, self.config)
        self.kpnn_vars = self.kpnn.setup_network(self.datasets[0][0], edges, outputs)
        
    def _load_config(self, config_path: str) -> dict:
        """
        Load the configuration from a YAML file.

        Args:
            config_path (str): The path to the configuration YAML file.

        Returns:
            dict: The configuration dictionary.
        """
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
            for key, value in config.items():
                if not isinstance(value, dict):
                    setattr(self, key, value)
        
        np.random.seed(self.seed)
        
        return config
        
    def _init_wandb(self, approach: str, config: dict) -> None:
        wandb.init(project="CPD", name=f'{approach.upper()}', entity="ethanmclark1")
        wandb.config.TPM = config['TPM']
        wandb.config.minmax = config['minmax']
        wandb.config.val_size = config['val_size']
        wandb.config.test_size = config['test_size']
        wandb.config.batch_size = config['batch_size']
        wandb.config.num_epochs = config['num_epochs']
        wandb.config.sma_window = config['sma_window']
        wandb.config.update(config[f'{approach}'])
        
    def _gather_data(self, input_file: str, edges_file: str, labels_file: str, config: dict) -> tuple:
        """
        Load and preprocess the input data, labels, and edge data.

        Args:
            input_file (str): The path to the input data file.
            edges_file (str): The path to the edge data file.
            labels_file (str): The path to the labels file.
            config (dict): The configuration dictionary.

        Returns:
            tuple: A tuple containing the edges, genes, and outputs.
        """
        
        self.data, genes_x, barcodes_x = DataPipeline.load_input_data(input_file, config)
        self.labels, barcodes = DataPipeline.load_data_labels(labels_file, barcodes_x, output_dir)
        edges, genes, genes_x = DataPipeline.load_edge_data(edges_file, genes_x, output_dir)
        
        outputs, self.data, self.labels = DataPipeline.validate_data(genes_x, edges, barcodes, barcodes_x, self.data, self.labels, output_dir)
        
        self.datasets = DataPipeline.generate_datasets(self.data, genes, genes_x, barcodes, self.labels, config)
        
        return edges, genes, outputs
    
    def prepare_data(self) -> tuple:
        train_x, train_Y = self.datasets[0]
        val_x, val_Y = self.datasets[1]
        test_x, test_Y = self.datasets[2]
        
        train_x = torch.tensor(train_x.toarray()).t().float()
        val_x = torch.tensor(val_x.toarray()).t().float()
        test_x = torch.tensor(test_x.toarray()).t().float()
        
        train_Y = torch.from_numpy(train_Y).float()
        val_Y = torch.from_numpy(val_Y).float()
        test_Y = torch.from_numpy(test_Y).float()
        
        train = TensorDataset(train_x, train_Y.view(-1))
        val = TensorDataset(val_x, val_Y.view(-1))
        test = TensorDataset(test_x, test_Y.view(-1))
        
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
        
    def train(self, name: str, train_loader: DataLoader, val_loader: DataLoader) -> None:
        approach = getattr(self, name)
        self._init_wandb(name, self.config)
        
        train_losses, train_aucs = [], []
        val_losses, val_aucs = [], []
        
        for epoch in range(self.num_epochs):
            train_loss = 0
            train_outputs, train_labels = [], []
            val_loss = 0
            val_outputs, val_labels = [], []
            
            for data, labels in train_loader:
                output = approach(data).view(-1)
                loss = approach.loss(output, labels)
                
                approach.optim.zero_grad()
                loss.backward()
                approach.optim.step()
                
                train_loss += loss.item()
                
                predicted_probs = torch.sigmoid(output)
                train_outputs.extend(predicted_probs.detach().cpu().numpy())
                train_labels.extend(labels.detach().cpu().numpy())
                
            with torch.no_grad():
                for data, labels in val_loader:
                    output = approach(data).view(-1)
                    loss = approach.loss(output, labels)
                    
                    val_loss += loss.item()
                    
                    predicted_probs = torch.sigmoid(output)
                    val_outputs.extend(predicted_probs.detach().cpu().numpy())
                    val_labels.extend(labels.detach().cpu().numpy())
                    
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_auc = roc_auc_score(train_labels, train_outputs)
            val_auc = roc_auc_score(val_labels, val_outputs)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_aucs.append(train_auc)
            val_aucs.append(val_auc)
            avg_train_loss = np.mean(val_losses[-self.sma_window:])
            avg_val_loss = np.mean(val_losses[-self.sma_window:])
            
            wandb.log({
                'Training Loss': avg_train_loss, 
                'Validation Loss': avg_val_loss,
                'Training AUC': train_auc,
                'Validation AUC': val_auc}, step=epoch)
            
    def test(self, name: str, test_loader: DataLoader) -> None:
        approach = getattr(self, name)
        approach.eval()
        
        test_loss, test_correct = 0, 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                output = approach(data).view(-1)
                loss = approach.loss(output, labels)
                
                test_loss += loss.item()
                predicted_probs = torch.sigmoid(output)
                predicted = (predicted_probs > 0.5).float()                
                test_correct += (predicted == labels).sum().item()
                
        test_loss /= len(test_loader)
        test_accuracy = test_correct / len(test_loader.dataset)
        
        wandb.log({
            'Test Loss': test_loss,
            'Test Accuracy': test_accuracy})
    
    def train_kpnn(self) -> None:
        """
        Train the KPNN model.
        """
        
        self._init_wandb('kpnn', self.config)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            train_x, train_Y = self.datasets[0]
            val_x, val_Y = self.datasets[1]
            
            val_x = val_x.toarray()
            y_val_weights = self.kpnn.get_weight_matrix(val_Y)

            for epoch in range(self.num_epochs):
                # Shuffle and split the training data into minibatches
                idxs = np.arange(train_x.shape[1])
                np.random.shuffle(idxs)
                minibatches = np.array_split(idxs, np.ceil(train_x.shape[1] / self.batch_size))

                for batch_idxs in minibatches:
                    # Prepare the training data for the current minibatch
                    train_x_batch = train_x[:, batch_idxs].toarray()
                    train_Y_batch = train_Y[:, batch_idxs]
                    y_batch_weights = self.kpnn.get_weight_matrix(train_Y_batch)

                    self._run_train_step(sess, train_x_batch, train_Y_batch, y_batch_weights)

                train_loss, train_auc = self._evaluate_model(sess, train_x_batch, train_Y_batch, y_batch_weights)
                val_loss, val_auc = self._evaluate_model(sess, val_x, val_Y, y_val_weights)
                
                wandb.log({
                    'Training Loss': train_loss,
                    'Training AUC': train_auc,
                    'Validation Loss': val_loss,
                    'Validation AUC': val_auc
                })
                    
            saver.save(sess, f'{self.model_dir}/model.ckpt', global_step=self.num_epochs)
                    
    def _run_train_step(self, sess: tf.Session, train_x_batch: np.ndarray, train_Y_batch: np.ndarray, y_batch_weights: np.ndarray) -> None:
        """
        Run a single training step.

        Args:
            sess (tf.Session): The TensorFlow session.
            train_x_batch (np.ndarray): The batch of input data.
            train_Y_batch (np.ndarray): The batch of labels.
            y_batch_weights (np.ndarray): The batch of label weights.
        """
        
        sess.run(self.kpnn_vars['trainer'], feed_dict={
            self.kpnn_vars['genes_orig']: train_x_batch,
            self.kpnn_vars['y_true']: train_Y_batch,
            self.kpnn_vars['y_weights']: y_batch_weights,
            self.kpnn_vars['node_dropout']: self.config['kpnn']['node_dropout'],
            self.kpnn_vars['gene_dropout']: self.config['kpnn']['gene_dropout']
        })

    def _evaluate_model(self, sess: tf.Session, x: np.ndarray, y_true: np.ndarray, y_weights: np.ndarray) -> tuple:
        """
        Evaluate the model on a given set of data.

        Args:
            sess (tf.Session): The TensorFlow session.
            x (np.ndarray): The input data.
            y_true (np.ndarray): The true labels.
            y_weights (np.ndarray): The label weights.

        Returns:
            tuple: A tuple containing the loss, and accuracy.
        """
        
        loss, y_hat = sess.run([self.kpnn_vars['loss'], self.kpnn_vars['y_hat']],
            feed_dict={
                self.kpnn_vars['genes_orig']: x,
                self.kpnn_vars['y_true']: y_true,
                self.kpnn_vars['y_weights']: y_weights,
                self.kpnn_vars['node_dropout']: self.config['kpnn']['node_dropout'],
                self.kpnn_vars['gene_dropout']: self.config['kpnn']['gene_dropout']
            }
        )
        
        y_true = y_true.flatten()
        y_hat = y_hat.flatten()
        auc = roc_auc_score(y_true, y_hat)
        return loss, auc
        
    def test_kpnn(self) -> None:
        """
        Test the trained KPNN model.
        """
        
        with tf.Session() as sess:
            latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
            saver = tf.train.Saver()
            saver.restore(sess, latest_checkpoint)

            test_x, test_Y = self.datasets[2]
            
            test_x = test_x.toarray()
            test_Y_weights = self.kpnn.get_weight_matrix(test_Y)

            test_loss, test_auc = self._evaluate_model(sess, test_x, test_Y, test_Y_weights)

            wandb.log({
                'Test Loss': test_loss,
                'Test AUC': test_auc
            })
            
        wandb.finish()
            
            
if __name__ == '__main__':
    input_data, edge_data, data_labels, output_dir = get_arguments()
    
    driver = Driver(input_data, edge_data, data_labels, output_dir)
    
    train_loader, val_loader, test_loader = driver.prepare_data()
    
    approaches = ['ann']
    for approach in approaches:
        driver.train(approach, train_loader, val_loader)
        driver.test(approach, test_loader)
        
    driver.train_kpnn()
    driver.test_kpnn()