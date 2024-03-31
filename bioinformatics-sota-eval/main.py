import os
import yaml
import wandb
import numpy as np
import tensorflow as tf

from arguments import get_arguments
from networks import ANN, GNN, PGNN, KPNN
from utils.data_pipeline import DataPipeline


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
        config = self._load_config(config_path)
        
        self._init_hyperparams(config)
        self._init_wandb(config)

        edges, genes, outputs = self._gather_data(input_file, edges_file, labels_file, config)
        
        self.ann = ANN(config)
        self.gnn = GNN(config)
        self.pgnn = PGNN(config)
        self.kpnn = KPNN(edges, genes, config)
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
        return config
    
    def _init_hyperparams(self, config: dict) -> None:
        """
        Initialize the hyperparameters from the configuration dictionary.

        Args:
            config (dict): The configuration dictionary.
        """
        
        self.seed = config['seed']
        np.random.seed(self.seed)
        self.alpha = config['hyperparams']['alpha']
        self.lambda_ = config['hyperparams']['lambda_']
        self.val_size = config['hyperparams']['val_size']
        self.num_epochs = config['hyperparams']['num_epochs']
        self.test_size = config['hyperparams']['test_size']
        self.batch_size = config['hyperparams']['batch_size']
        self.tpm_normalization = config['normalization']['TPM']
        self.gene_dropout = config['hyperparams']['gene_dropout']
        self.node_dropout = config['hyperparams']['node_dropout']
        self.num_grad_epsilon = config['hyperparams']['num_grad_epsilon']
        
    def _init_wandb(self, approach: str, config: dict) -> None:
        wandb.init(project="CPD", name=f'{approach}', entity="ethanmclark1")
        wandb.config.update(config['batch_size'])
        wandb.config.update(config['num_epochs'])
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
    
    def train(self) -> None:
        pass
    
    def test(self) -> None:
        pass
    
    def _train_kpnn(self) -> None:
        """
        Train the KPNN model.
        """
        
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

                if epoch % 10 == 0:
                    train_loss, train_error, train_accuracy = self._evaluate_model(sess, train_x_batch, train_Y_batch, y_batch_weights)
                    val_loss, val_error, val_accuracy = self._evaluate_model(sess, val_x, val_Y, y_val_weights)
                    
                    train_error = train_error.mean()
                    val_error = val_error.mean()
                    
                    saver.save(sess, f'{self.model_dir}/model.ckpt', global_step=epoch)

                    wandb.log({
                        'Training Loss': train_loss,
                        'Training Error': train_error,
                        'Training Accuracy': train_accuracy,
                        'Validation Loss': val_loss,
                        'Validation Error': val_error,
                        'Validation Accuracy': val_accuracy
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
            self.kpnn_vars['node_dropout']: self.node_dropout,
            self.kpnn_vars['gene_dropout']: self.gene_dropout
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
            tuple: A tuple containing the loss, error, and accuracy.
        """
        
        loss, y_hat, error, accuracy = sess.run(
            [self.kpnn_vars['loss'], self.kpnn_vars['y_hat'], self.kpnn_vars['error'], self.kpnn_vars['accuracy']],
            feed_dict={
                self.kpnn_vars['genes_orig']: x,
                self.kpnn_vars['y_true']: y_true,
                self.kpnn_vars['y_weights']: y_weights,
                self.kpnn_vars['node_dropout']: self.node_dropout,
                self.kpnn_vars['gene_dropout']: self.gene_dropout
            }
        )
        return loss, error, accuracy
        
    def _test_kpnn(self) -> None:
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

            test_loss, test_error, test_accuracy = self._evaluate_model(sess, test_x, test_Y, test_Y_weights)

            wandb.log({
                'Test Loss': test_loss,
                'Test Error': test_error,
                'Test Accuracy': test_accuracy
            })
            
            
if __name__ == '__main__':
    input_data, edge_data, data_labels, output_dir = get_arguments()
    
    driver = Driver(input_data, edge_data, data_labels, output_dir)
    
    driver.train()
    driver.test()