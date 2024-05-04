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
