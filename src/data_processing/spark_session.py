"""
Spark Session Configuration Module
===================================
Creates and configures a SparkSession for the Healthcare Intelligence System.
Handles Windows-specific environment setup for Java, Hadoop, and PySpark.
"""

import os
import sys
from pyspark.sql import SparkSession


def create_spark_session(config=None):
    """
    Create and configure a SparkSession for the Healthcare Intelligence System.

    Parameters
    ----------
    config : dict, optional
        Override dictionary with keys such as:
            - app_name       (str)  : Spark application name
            - master         (str)  : Spark master URL
            - driver_memory  (str)  : Driver memory allocation
            - java_home      (str)  : Path to JAVA_HOME
            - hadoop_home    (str)  : Path to HADOOP_HOME
            - pyspark_python (str)  : Path to Python interpreter for PySpark
        Any Spark config key (e.g. "spark.sql.shuffle.partitions") can also
        be passed and will be forwarded to the builder.

    Returns
    -------
    pyspark.sql.SparkSession
        A fully configured SparkSession instance.
    """
    if config is None:
        config = {}

    # ------------------------------------------------------------------ #
    # 1. Windows environment variables                                    #
    # ------------------------------------------------------------------ #
    os.environ["JAVA_HOME"] = os.environ.get(
        "JAVA_HOME", config.get("java_home", r"C:\Program Files\Java\jdk-11")
    )
    os.environ["HADOOP_HOME"] = os.environ.get(
        "HADOOP_HOME", config.get("hadoop_home", r"C:\hadoop")
    )
    os.environ["PYSPARK_PYTHON"] = os.environ.get(
        "PYSPARK_PYTHON", config.get("pyspark_python", sys.executable)
    )

    # Ensure winutils can be found on Windows
    hadoop_bin = os.path.join(os.environ["HADOOP_HOME"], "bin")
    if hadoop_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

    # ------------------------------------------------------------------ #
    # 2. Resolve core settings (allow overrides)                          #
    # ------------------------------------------------------------------ #
    app_name = config.get("app_name", "HealthcareIntelligenceSystem")
    master = config.get("master", "local[*]")
    driver_memory = config.get("driver_memory", "8g")

    # ------------------------------------------------------------------ #
    # 3. Build SparkSession                                               #
    # ------------------------------------------------------------------ #
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    )

    # Forward any extra Spark config keys supplied by the caller
    reserved_keys = {
        "app_name", "master", "driver_memory",
        "java_home", "hadoop_home", "pyspark_python",
    }
    for key, value in config.items():
        if key not in reserved_keys:
            builder = builder.config(key, str(value))

    spark = builder.getOrCreate()

    # ------------------------------------------------------------------ #
    # 4. Reduce console noise                                             #
    # ------------------------------------------------------------------ #
    spark.sparkContext.setLogLevel("WARN")

    return spark
