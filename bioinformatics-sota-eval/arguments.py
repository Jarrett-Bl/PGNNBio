import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Cancer Detection using a Knowledge-Primed Neural Network')
    
    parser.add_argument(
        '--input_data',
        type=str,
        help='Path to the input data file (.csv or .h5)', default='data/tcr_data.h5',
    )
    
    parser.add_argument(
        '--edge_data',
        type=str,
        help='Path to the input edges file (.csv or .h5)', default='data/tcr_edge_lst.csv',
    )
    
    parser.add_argument(
        '--data_labels',
        type=str,
        help='Path to the input labels file (.csv or .h5)', default='data/tcr_class_labels.csv',
    )
    
    parser.add_argument(
        '--database',
        type=str,
        choices=['kegg', 'hallmark', 'wiki_pathways'],
        help='Structure for relations data', default='wiki_pathways',
    )
    
    parser.add_argument(
        '--model',
        type=str,
        nargs='+',
        help='Models to use',
        default=['pgnn'],
    )
    
    parser.add_argument(
        '--pathway_importance_type',
        type=str,
        choices=['naive', 'shap', 'lime'],
        help='Method to calculate feature importance', default='shap',
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the output directory', default='data/tmp/',
    )
    
    args = parser.parse_args()
    
    return args.input_data, args.edge_data, args.data_labels, args.database, args.model, args.pathway_importance_type, args.output_dir