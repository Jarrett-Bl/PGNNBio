
# Bioinformatics state-of-the-art Evaluation

This repository aims to evaluate the efficacy of various neural network architectures, including Knowledge-Primed Neural Network (KPNN), Pathway-Guided Neural Network (PGNN), Graph Neural Network (GNN), and Artificial Neural Network (ANN), in predicting T-cell receptor (TCR) stimulation.

## Overview

The project focuses on analyzing gene expression data and leveraging prior biological knowledge to enhance the predictive power of neural networks. By incorporating gene hierarchies, pathway information, and graph structures, we aim to improve the performance of models in identifying TCR stimulation patterns.

## Dataset

The dataset used in this study consists of gene expression data, gene hierarchy information (in the form of edges), and class labels for TCR stimulation. The dataset is provided in HDF5 and CSV formats.

## Models

The following neural network architectures are implemented and evaluated:

1. **Knowledge-Primed Neural Network (KPNN)**: This model incorporates prior knowledge about gene hierarchies and regulatory relationships into the network structure. By leveraging the hierarchical structure of the data, KPNN aims to enhance the interpretability and performance of the model.
2. **Pathway-Guided Neural Network (PGNN)**: PGNN utilizes pathway information to guide the learning process. By incorporating knowledge about biological pathways, this model aims to capture the underlying biological mechanisms influencing TCR stimulation.
3. **Graph Neural Network (GNN)**: GNN represents the T-Cell Receptor pathway as a graph, where nodes represent genes, and edges represent interactions or relationships between genes. By leveraging the graph structure, GNN can simulate patterns and dependencies within the data.
4. **Artificial Neural Network (ANN)**: ANN serves as a baseline model, consisting of fully connected layers without incorporating any prior knowledge or structured representations.

## Usage

To run the code and evaluate the models, follow these steps:

1. Clone the repository:

   ```
   $ git clone https://github.com/ethanmclark1/bioinformatics-sota-eval.git
   $ cd bioinformatics-sota-eval
   ```
2. Install the required dependencies

   ```
   $ pip install -r requirements.txt
   ```
3. Run the main script:

   ```
   $ python main.py --input_data bioinformatics-sota-eval/data/tcr_data.h5 --edge_data bioinfomatics-sota-eval/data/tcr_edge_lst.csv --data_labels bioinformatics-sota-eval/data/tcr_class_labels.csv --output_dir bioinformatics-sota-eval/data/tmp
   ```

The script will train and evaluate each model on the provided dataset, reporting the performance metrics (i.e., loss, AUC) for training, validation, and test sets. Additionally, the trained models will be saved in the specified output directory.

## Contributing

Contributions to this repository are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Acknowledgments

We would like to acknowledge the contributors and authors of the datasets and libraries used in this project.

**TODO**
