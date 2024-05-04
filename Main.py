from parameters import parameter_parser
from SSNTasks import SNNTasks
from TemporalData import TemporalData
import logging.config
import SSNUtils as utils

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('default')

if __name__ == "__main__":
    # Read parameters for experiments
    args = parameter_parser()
    # Load the temporal data
    temporal_data = TemporalData(args.input_path)
    samples, targets = temporal_data.load(display=True)
    num_inputs = temporal_data.get_n_inputs()
    # Create a new task object
    task = SNNTasks()

    if args.task == "train":
        task.train(num_inputs=num_inputs, samples=samples, targets=targets, args=args)
    elif args.task == "recall":
        task.recall(num_inputs=num_inputs, samples=samples, targets=targets, args=args)
    elif args.task == "transfer":
        task.transfer(num_inputs=num_inputs, samples=samples, targets=targets, args=args)

    # Plots
    # class_labels=["r_1", "r_2", "r_3", "r_4", "r_5", "r_6", "r_7"]
    # model_file="D:/DataSets/Pigeons/Transfer learning/Experiments/Jason & Davison/Cond1/py_results no seed/Models/results.csv"
    # Plot raster plot
    # utils.plot_raster(model_file=model_file, samples=samples, targets=targets, num_inputs=num_inputs, model_ids=[0],
    #                   class_labels=class_labels)
    # # Plot firing rates distribution per class
    # utils.plot_firing_distribution(model_file=model_file, samples=samples, num_inputs=num_inputs,  model_ids=[0],
    #                                class_labels=class_labels)
    # # utils.plot_performance_evolution(model_file=model_file, mean=True)
    # #
    # utils.plot_confusion_matrix(model_file=model_file, samples=samples, targets=targets, num_inputs=num_inputs,
    #                             model_ids=[0], class_labels=class_labels, figsize=(8, 6), text_size=8)
    #
    #
    # utils.plots_show()

    #
    # utils.save_firing_rates(model_file=model_file, samples=samples, targets=targets, num_inputs=num_inputs,
    #                        model_ids=[0], class_labels=class_labels)

