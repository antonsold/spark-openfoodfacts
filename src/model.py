from pyspark.sql import SparkSession, DataFrame
import sys
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from logger import Logger
import traceback
from typing import List, Tuple
import yaml

SHOW_LOG = True


class Model:
    def __init__(self, config_path: str) -> None:
        self.log = Logger(SHOW_LOG).get_logger(__name__)
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except:
            self.log.error("Unable to load config")
            sys.exit(1)

        try:
            mongo_input_string = \
                f"mongodb://{self.config['mongo']['host']}/{self.config['mongo']['input']['db']}.{self.config['mongo']['input']['collection']}"
            mongo_output_string = \
                f"mongodb://{self.config['mongo']['host']}/{self.config['mongo']['output']['db']}.{self.config['mongo']['output']['collection']}"

            self.spark = (
                SparkSession
                .builder
                .master(self.config["spark"]["master"])
                .appName(self.config["spark"]["app_name"])
                .config("spark.mongodb.input.uri", mongo_input_string)
                .config("spark.mongodb.output.uri", mongo_output_string)
                .config("spark.jars.packages", self.config["mongo"]["package"])
                .getOrCreate()
            )
        except:
            traceback.format_exc()
            self.log.error("Unable to create Spark Session. Check configuration file")
            sys.exit(1)
        try:
            self.n_samples = self.config['n_samples']
            self.n_clusters = self.config['n_clusters']
            self.random_seed = self.config['random_seed']
            self.test_size = self.config['test_size']
            self.input_fields = self.config["mongo"]["input"]["fields"]
            self.log.info("Model initialized.")
        except:
            self.log.error("Unable to load model parameters. Check configuration file")
            sys.exit(1)
        self.data = None
        self.pipeline = None

    def load_data(self) -> bool:
        try:
            self.data = (
                self.spark
                .read
                .format("mongo")
                .load()
                .limit(self.n_samples)
                .select(self.input_fields)
            )
            return True
        except:
            self.log.error(traceback.format_exc())
            return False

    def split_data(self) -> Tuple[DataFrame, DataFrame]:
        try:
            train, test = (
                self
                .data
                .randomSplit(
                    weights=[1 - self.test_size, self.test_size],
                    seed=self.random_seed
                )
            )
        except:
            self.log.error("Unable to split data")
            sys.exit(1)
        return train, test

    @staticmethod
    def drop_null_columns(df: DataFrame, thresh: int) -> DataFrame:
        null_counts = (
            df
            .select([
                F.count(
                    F.when(F.col(c).isNull(), c)
                )
                .alias(c)
                for c in df.columns
            ])
            .collect()[0].asDict()
        )
        columns_to_drop = [name for name, count in null_counts.items() if count >= thresh]
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
            self.data = Model.drop_null_columns(self.data, thresh)
            self.data = Model.drop_non_numeric_columns(self.data, ["_id", "product_name"])
            input_cols = [col for col in self.data.columns if col not in ["_id", "product_name"]]
            self.pipeline = Pipeline(stages=[
                Imputer(inputCols=input_cols, outputCols=input_cols),
                VectorAssembler(inputCols=input_cols, outputCol="vect"),
                StandardScaler(withMean=True, inputCol="vect", outputCol="norm"),
                KMeans(featuresCol="norm", predictionCol="pred", k=self.n_clusters, seed=self.random_seed)
            ])
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
        data_to_write = (
            data
            .withColumn("timestamp", F.current_timestamp())
            .select([
                F.col("_id").alias("product_id"),
                "product_name",
                "timestamp",
                vector_to_array(F.col("norm")).alias("features"),
                "pred"
            ])
        )
        try:
            (
                data_to_write
                .write
                .format("mongo")
                .mode("overwrite")
                .save()
            )
            return True
        except:
            self.log.error(traceback.format_exc())
            return False