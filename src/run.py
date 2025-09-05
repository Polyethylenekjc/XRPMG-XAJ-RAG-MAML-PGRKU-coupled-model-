from src.utils.Logger import Logger
import yaml
from src.pipeline import MLExperimentPipeline
import os

def main():
    # Create the log directory if it doesn't exist
    log_file_path = "res/logs/experiment.log"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Initialize the logging system
    Logger.initialize(log_file=log_file_path)
    Logger.set_level("DEBUG")  # Set the logging level to DEBUG

    # Load the configuration file
    with open('Config/test.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Create and run the experiment pipeline
    pipeline = MLExperimentPipeline(config)
    pipeline.runExperiment()

if __name__ == "__main__":
    main()
