import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="PigeonsSSN")

    parser.add_argument('--task',
                        default="train",
                        type=str,
                        help="The task to perform:"
                             "1) train: create and train a new SSN model; "
                             "2) recall: classify data using a pretrained SSN model,"
                             "3) transfer: apply the knowledge gained in a pretrained model to classify a different "
                             "dataset. The transfer_type argument defines the type of transfer learning either "
                             "feature extraction of fine tuning.")

    parser.add_argument('--input_path',
                        default="D:/DataSets/Pigeons/csv_samples/",
                        type=str, help="The directory that contains the data sample files (sam*.csv) and the file "
                                       "tar_class.csv'' having their class labels")
    parser.add_argument('--output_path',
                        # Landon & Davison
                        default="D:/DataSets/Pigeons/csv_samples/output/",
                        type=str,
                        help="The main directory to save the results of the experiments. These directory must contain "
                             "two directories:'feature_extraction' and 'fine_tuning'. Additionally, the 'fine_tuning' "
                             "directory must contain three more directories: 'bounded', 'normal' and 'uniform'.")

    # Nested cross validation parameters
    # The inner and outer splits are stratified k-fold cross validation strategies
    parser.add_argument('--outer_splits',
                        default=10,
                        type=int,
                        help="The number of splits of the outer loop of the nested cross-validation.")
    parser.add_argument('--inner_splits',
                        default=5,
                        type=int,
                        help="The number of splits of the inner loop of the nested cross-validation.")
    parser.add_argument('--nested_seed',
                        default=None,
                        type=int,
                        help="The seed generator for shuffling and splitting the data in both outer and inner loops"
                             "of the nested cross validation.")

    # Training Differential Evolution. These parameters are also used in fine tuning approach of transfer learning
    parser.add_argument('--pop_size',
                        default=10,
                        type=int,
                        help="A multiplier for setting the total population size. The population has "
                             "pop_size * d individuals, where d is the dimensionality of an individual (agent or "
                             "candidate solution). Pop_size must be less than the number of rows in the specified "
                             "column")
    parser.add_argument('--max_iterations',
                        default=200
                        type=int,
                        help="The maximum number of iterations of the Differential Evolution")

    # Recall and transfer learning model arguments
    parser.add_argument('--model_file',
                        default="D:/DataSets/Pigeons/Models/results.csv",
                        type=str, help="The results of a previously training model to use for recall or transfer learning")
    parser.add_argument('--model_ids',
                        default=None, # [1,2] All models (10 models since the outer loop of the nested cross validation was set to 10)
                        type=int,
                        help="The model id to transfer from the list of models stored in the model_file. If None, it will run all the models in the model_file.")
    parser.add_argument('--param_column',
                        default="best_parameters",
                        type=str,
                        help="Column that has the parameters of the single SNN.")
    parser.add_argument('--firings_column',
                        default="best_firing_rates",
                        type=str,
                        help="Column that has the average firing rates for each class.")

    # Transfer learning arguments
    parser.add_argument('--transfer_type',
                        default="fine_tuning",
                        type=str, help="The transfer learning approach ('feature_extraction' or 'fine_tuning'. "
                                       "feature_extraction: take the underlying patterns (also called weights) a "
                                       "pretrained model has learned and adjust its outputs (firing rates) to be more "
                                       "suited to your problem. "
                                       "fine_tuning: take the underlying patterns (also called weights) of a "
                                       "pretrained model and adjust (fine-tune) them to your own problem")

    #  Feature extraction approach
    parser.add_argument('--cv',
                        default=True,
                        type=bool, help="Activate the stratified cross validation if the approach is "
                                        "'feature_extraction' otherwise the firing rates will be calculated using the "
                                        "whole dataset.")
    parser.add_argument('--cv_num_splits',
                        default=5,
                        type=int,
                        help="If the cv=True, this parameter indicates the number of splits of the stratified k-fold "
                             "cross validation.")
    parser.add_argument('--cv_seed',
                        default=None,
                        type=int,
                        help="The seed generator for shuffling and splitting the data in the cross validation.")
    parser.add_argument('--cv_scoring',
                        default='accuracy',
                        type=str,
                        help="Strategy to evaluate the performance of the cross-validated model on the test set. See"
                             "https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for the "
                             "most common use cases,")

    # Fine tuning (retraining)
    parser.add_argument('--pop_dist',
                        default='uniform',
                        type=str,
                        help="The probability distribution strategy to create the initial population of the "
                             "Differential Evolution for retraining the model. This strategy should be one of: "
                             "'uniform', 'bounded, 'normal'")
    parser.add_argument('--min_val',
                        default=0,
                        type=float,
                        help="An scalar or array indicating the minimum value of each dimension of a candidate "
                             "solution of the Differential Evolution. Used when pop_dist='bounded'")
    parser.add_argument('--max_val',
                        default=1.0,
                        type=float,
                        help="A scalar or array indicating the maximum value of each dimension of a candidate "
                             "solution of the Differential Evolution.Used when pop_dist='bounded'")
    parser.add_argument('--std_dev',
                        default=0.5,
                        type=float,
                        help="A scalar or array indicating the standard deviation of each dimension of a candidate "
                             "solution of the Differential Evolution. Used when pop_dist='normal'")
    parser.add_argument('--pop_seed',
                        default=None,
                        type=int,
                        help="The seed generator for creating a random population")


    return parser.parse_args()
