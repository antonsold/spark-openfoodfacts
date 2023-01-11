import os
import sys

from model import Model

CONFIG_PATH = os.path.join(os.getcwd(), "config.yaml")


class Trainer:
    def __init__(self, config_path):
        self.config_path = config_path

    def execute(self):
        model = Model(self.config_path)
        read_success = model.load_data()
        if not read_success:
            model.log.error("Error reading data")
            sys.exit(1)
        else:
            model.log.info("Finished loading data")
        model.fit()
        predictions = model.predict(model.data)
        write_success = model.write(predictions)
        if write_success:
            model.log.info(f"Pipeline executed successfully.")
        else:
            model.log.info("Error writing data")
        model.spark.stop()


if __name__ == "__main__":
    Trainer(CONFIG_PATH).execute()
