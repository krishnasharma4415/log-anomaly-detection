from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import os

def create_spark_session(app_name="LogAnomalyDetection"):
    """
    Create optimized Spark session for single-machine deployment
    """
    conf = SparkConf().setAll([
        ("spark.master", "local[*]"),
        ("spark.driver.memory", "10g"),
        ("spark.driver.maxResultSize", "4g"),
        ("spark.sql.adaptive.enabled", "true"),
        ("spark.sql.adaptive.coalescePartitions.enabled", "true"),
        ("spark.sql.execution.arrow.pyspark.enabled", "true"),
        ("spark.serializer", "org.apache.spark.serializer.KryoSerializer"),
        ("spark.sql.repl.eagerEval.enabled", "true"),
        ("spark.sql.repl.eagerEval.maxNumRows", 20)
    ])
    
    spark = SparkSession.builder \
        .config(conf=conf) \
        .appName(app_name) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

def optimize_for_local_processing(spark):
    """
    Additional optimizations for local development
    """
    spark.conf.set("spark.sql.shuffle.partitions", "8")
    
    return spark