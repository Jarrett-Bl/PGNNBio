# Bioinformatics state-of-the-art Evaluation

This repository aims to evaluate the efficacy of state-of-the-art neural network architectures in the field of bioinformatics for predicting T-cell receptor (TCR) stimulation using single-cell RNA sequencing (scRNA-seq) data. The architectures under evaluation include:

1. Knowledge-Primed Neural Network (KPNN): A sparsely connected feed-forward neural network that incorporates prior knowledge about gene hierarchies and regulatory relationships, ensuring high interpretability by mirroring the biological pathway structure.
2. Pathway-Guided Neural Network (PGNN): A neural network that utilizes a "pathway" layer to aggregate information from multiple biological pathways, including the TCR pathway, and feeds the computed pathway scores into a densely connected artificial neural network for prediction.
3. Graph Neural Network (GNN): A graph-based model that represents the TCR pathway as a graph, with nodes corresponding to genes and edges representing gene interactions. The GNN employs a message-passing framework and graph convolutional network (GCN) architecture to capture complex dependencies within the pathway.
4. Artificial Neural Network (ANN): A baseline model consisting of fully connected layers, which learns to predict TCR stimulation directly from gene expression data without incorporating prior biological knowledge or structured representations.

By comparing the performance and characteristics of these biologically-informed neural network architectures, this repository aims to provide insights into their strengths and weaknesses in the context of prediction, ultimately guiding future research on developing effective composite models that leverage the advantages of each approach.

## Dataset

The dataset used in this study consists of single-cell RNA sequencing (scRNA-seq) data, along with corresponding biological knowledge in the form of gene hierarchies, pathway information, and graph structures. The scRNA-seq data provides gene expression measurements at the individual cell level. The biological information is used to guide and inform the creation of the neural network architectures, incorporating the relationships and dependencies between genes and pathways. The dataset also includes class labels indicating the presence or absence of TCR stimulation for each cell.

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
   $ python main.py --input_data data/tcr_data.h5 --edge_data data/tcr_edge_lst.csv --data_labels data/tcr_class_labels.csv --output_dir data/tmp
   ```

## Contributing

Contributions to this repository are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Paper Citation

If you found this repository helpful, consider citing the following papers:

<pre>
@article{fortelny2020knowledge,
  title={Knowledge-primed neural networks enable biologically interpretable deep learning on single-cell sequencing data},
  author={Fortelny, Nikolaus and Bock, Christoph},
  journal={Genome biology},
  volume={21},
  pages={1--36},
  year={2020},
  publisher={Springer}
}
</pre>

<pre>@article{deng2020pathway, 
  title={Pathway-Guided Deep Neural Network toward Interpretable and Predictive Modeling of Drug Sensitivity}, 
  author={Deng, Liang and Cai, Yidong and Zhang, Wenbo and Yang, Woosung and Gao, Bo and Liu, Haibo}, 
  journal={Journal of Chemical Information and Modeling}, 
  volume={60}, 
  number={10}, 
  pages={4497--4505}, 
  year={2020}, 
  publisher={American Chemical Society}, 
  doi={10.1021/acs.jcim.0c00331}, 
  pmid={32804489}
}
</pre>
