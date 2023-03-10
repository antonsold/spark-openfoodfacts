import sys
import json
import traceback
from typing import List, Tuple
from dataloader import DataLoader
import yaml
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from logger import Logger

SHOW_LOG = True


class Model:
    def __init__(self, config_path: str, credentials_path) -> None:
        self.log = Logger(SHOW_LOG).get_logger(__name__)
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except:
            self.log.error("Unable to load config")
            sys.exit(1)
        try:
            with open(credentials_path, "r") as f:
                credentials = json.load(f)
        except:
            self.log.error("Unable to load credentials")
            sys.exit(1)
        try:
            self.dataloader = DataLoader(self.config["spark"], self.config["mongo"], credentials)
        except:
            self.log.error(traceback.format_exc())
            self.log.error("Unable to create Spark Session. Check configuration file")
            sys.exit(1)
        try:
            self.n_samples = self.config["n_samples"]
            self.n_clusters = self.config["n_clusters"]
            self.random_seed = self.config["random_seed"]
            self.test_size = self.config["test_size"]
            self.log.info("Model initialized.")
        except:
            self.log.error("Unable to load model parameters. Check configuration file")
            sys.exit(1)
        self.data = None
        self.pipeline = None

    def load_data(self) -> bool:
        try:
            self.data = self.dataloader.read(self.n_samples)
            return True
        except:
            self.log.error(traceback.format_exc())
            return False

    def split_data(self) -> Tuple[DataFrame, DataFrame]:
        try:
            train, test = self.data.randomSplit(
                weights=[1 - self.test_size, self.test_size], seed=self.random_seed
            )
        except:
            self.log.error("Unable to split data")
            sys.exit(1)
        return train, test

    @staticmethod
    def drop_null_columns(df: DataFrame, thresh: int) -> DataFrame:
        null_counts = (
            df.select(
                [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
            )
            .collect()[0]
            .asDict()
        )
        columns_to_drop = [
            name for name, count in null_counts.items() if count >= thresh
        ]
        df = df.drop(*columns_to_drop)
        return df

    @staticmethod
    def drop_non_numeric_columns(df: DataFrame, exclude: List[str]):
        non_num_cols = [
            f.name
            for f in df.schema.fields
            if not isinstance(f.dataType, DoubleType)
            and not isinstance(f.dataType, IntegerType)
            and f.name not in exclude
        ]
        df = df.drop(*non_num_cols)
        return df

    def fit(self) -> None:
        try:
            thresh = 0.5 * self.n_samples
            # Dropping columns with >50% null values and non-numeric columns
            self.data = Model.drop_null_columns(self.data, thresh)
            self.data = Model.drop_non_numeric_columns(
                self.data, ["_id", "product_name"]
            )
            input_cols = [
                col for col in self.data.columns if col not in ["_id", "product_name"]
            ]

            # Imputing missing fields with mean, then applying standard scaling
            self.pipeline = Pipeline(
                stages=[
                    Imputer(inputCols=input_cols, outputCols=input_cols),
                    VectorAssembler(inputCols=input_cols, outputCol="vect"),
                    StandardScaler(withMean=True, inputCol="vect", outputCol="norm"),
                    KMeans(
                        featuresCol="norm",
                        predictionCol="pred",
                        k=self.n_clusters,
                        seed=self.random_seed,
                    ),
                ]
            )
            self.log.info("Pipeline initialized successfully.")
        except:
            self.log.error(traceback.format_exc())
            self.log.error("Unable to initialize pipeline")
            sys.exit(1)

        self.train, self.test = self.split_data()

        try:
            self.model = self.pipeline.fit(self.train)
            self.log.info("Trained KMeans model")
        except:
            self.log.error(traceback.format_exc())
            sys.exit(1)

    def predict(self, data: DataFrame) -> DataFrame:
        try:
            predictions = self.model.transform(data)
            self.log.info("Generated predictions.")
        except:
            self.log.error(traceback.format_exc())
            sys.exit(-1)
        return predictions

    def write(self, data: DataFrame) -> bool:
        data_to_write = data.withColumn("timestamp", F.current_timestamp()).select(
            [
                F.col("_id").alias("product_id"),
                "product_name",
                "timestamp",
                vector_to_array(F.col("norm")).alias("features"),
                "pred",
            ]
        )
        try:
            self.dataloader.write(data_to_write)
            return True
        except:
            self.log.error(traceback.format_exc())
            return False
