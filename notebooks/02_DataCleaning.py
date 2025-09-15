# Databricks notebook source

# COMMAND ----------
from loguru import logger
import yaml
import sys
from pyspark.sql import SparkSession
import pandas as pd
import datetime

from mlops_course.config import ProjectConfig
from mlops_course.data_processor import DataProcessor

config = ProjectConfig.from_yaml("../project_config.yml", "dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))
# COMMAND ----------
# Set up Spark session for Databricks Connect
spark = SparkSession.builder \
    .getOrCreate()

# COMMAND ----------
data_processor = DataProcessor(config, "raw_bike_trips", spark)

# COMMAND ----------

# Read the data
logger.info("Reading data from Databricks tables...")
df = data_processor.read_data(config.data_parameters["min_date"], config.data_parameters["max_date"])

logger.info("Preprocessing data...")
df = data_processor.preprocess_data(df)

logger.info("Getting aggregated data (+ filling missing slots)...")
pdf = data_processor.get_agg_data(df)

pdf.head()

# COMMAND ----------
logger.info("Transforming time-series data into features and targets...")
pdf = data_processor.transform_ts_data_into_features_and_target(pdf, 
                                                                config.data_parameters["input_seq_len"], 
                                                                config.data_parameters["step_size"])

pdf.head()
# COMMAND ----------
# Split the data into training and testing sets
logger.info("Splitting data into training and testing sets...")
train_df, test_df = data_processor.split_data(pdf, config.data_parameters["cutoff_date"])

# COMMAND ----------
# Save the data into Databricks tables
logger.info("Saving data into Databricks tables...")
data_processor.save_to_catalog(train_df, test_df)

# Enable change data feed (only once!)
logger.info("Enabling change data feed...")
data_processor.enable_change_data_feed()

# COMMAND ----------