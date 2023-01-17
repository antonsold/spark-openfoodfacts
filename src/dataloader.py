from typing import Dict
from pyspark.sql import DataFrame, SparkSession


class DataLoader:
    def __init__(self, spark_info: Dict[str, str], conn_info: dict, credentials: Dict[str, str]):
        host = conn_info['host']
        user = credentials["username"]
        password = credentials["password"]
        mongo_input_string = f"mongodb://{user}:{password}@{host}/{conn_info['input']['db']}.{conn_info['input']['collection']}?authSource=admin"
        mongo_output_string = f"mongodb://{user}:{password}@{host}/{conn_info['output']['db']}.{conn_info['output']['collection']}?authSource=admin"
        self.spark = (
            SparkSession.builder.master(spark_info["master"])
            .appName(spark_info["app_name"])
            .config("spark.mongodb.input.uri", mongo_input_string)
            .config("spark.mongodb.output.uri", mongo_output_string)
            .config("spark.jars.packages", conn_info["package"])
            .getOrCreate()
        )
        self.input_fields = conn_info["input"]["fields"]

    def read(self, n_samples) -> DataFrame:
        data = (
            self.spark.read.format("mongo")
            .load()
            .limit(n_samples)
            .select(self.input_fields)
        )
        return data

    def write(self, data):
        data.write.format("mongo").mode("overwrite").save()

    def shutdown(self):
        self.spark.stop()
