"""Data preprocessing module."""

import datetime
import time

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from pyspark.sql import functions as F

from sklearn.model_selection import train_test_split

from mlops_course.config import ProjectConfig
from tqdm import tqdm


class DataProcessor:
    """Data preprocessing class."""

    def __init__(self, config: ProjectConfig, table_name: str, spark: SparkSession) -> None:
        """Initialize the data processor."""
        self.config = config
        self.table_name = table_name
        self.spark = spark

    def read_data(self, min_date: str, max_date: str) -> SparkDataFrame:
        """Read the data from the table."""
        columns = ["Ciclo_Estacion_Retiro", "Fecha_Retiro", "Hora_Retiro"]
        # Read table, select columns and filter by date
        df = (self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.{self.table_name}")
                    .select(columns)
                    .filter(F.col('Fecha_Retiro').between(min_date, max_date))
                    )
        return df


    def preprocess_data(self, df: SparkDataFrame) -> SparkDataFrame:
        """Preprocess the data."""
        # Rename the columns
        df = df.withColumnRenamed("Ciclo_Estacion_Retiro", "StationID")
        df = df.withColumnRenamed("Fecha_Retiro", "Date")
        df = df.withColumnRenamed("Hora_Retiro", "Hour")

        # There is something wrong with the data, the HourDate is a datetime, but the Date is wrong.
        # For example, when Date is 2025-06-01, the Hour is 2025-09-05 12:20:10
        # The Hour should only be 12:20:10, the 2025-09-05 should not be there.
        # Let's do the following:
        # 1. Combine Date and Hour into a single datetime column (e.g. 2025-06-01 12:20:10) - Call this column "PickUpDateTime"
        #   - Convert Date and Hour to strings
        #   - Concatenate them
        #   - Convert to datetime
        # 2. Round the datetime to the nearest hour
        # 3. Convert the datetime to a date

        # Convert Date and Hour to strings
        df = df.withColumn("Date", F.col("Date").cast("string"))
        df = df.withColumn("Hour", F.col("Hour").cast("string"))

        # Concatenate them, but for Hour we only want the time part (e.g. 12:20:10), only the last 8 characters
        df = df.withColumn("PickUpDateTime", F.concat(F.col("Date"), F.substring(F.col("Hour"), -9, 9)))

        # Convert PickUpDateTime to datetime
        df = df.withColumn("PickUpDateTime", F.to_timestamp("PickUpDateTime"))

        # Floor the datetime to the nearest hour
        df = df.withColumn("PickUpDateTime", F.date_trunc("hour", F.col("PickUpDateTime")))

        # Keep only the columns we need
        df = df.select("StationID", "PickUpDateTime")

        return df

    def get_agg_data(self, df: SparkDataFrame) -> pd.DataFrame:
        """Get the aggregated data."""
        # Count the number of rides
        df = df.groupby(["StationID", "PickUpDateTime"]).count().withColumnRenamed("count", "Rides")

        # Sort by Rides descending
        df = df.sort(F.desc("Rides"))

        # Convert to pandas
        df = df.toPandas()

        # Add missing slots
        df = self.add_missing_slots(df)

        return df

    def add_missing_slots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the missing slots of time."""
        location_ids = df['StationID'].unique()
        full_range = pd.date_range(df['PickUpDateTime'].min(), df['PickUpDateTime'].max(), freq='h')
        output = pd.DataFrame()
        for location_id in tqdm(location_ids):
            location_df = df[df['StationID'] == location_id]
            location_df = location_df.set_index('PickUpDateTime')
            location_df.index = pd.DatetimeIndex(location_df.index)
            location_df = location_df.reindex(full_range, fill_value=0)
            location_df['StationID'] = location_id
            output = pd.concat([output, location_df])
        
        return output.reset_index(names=['Date'])

    def get_cutoff_indices(self, df: pd.DataFrame, input_seq_len: int, step_size: int) -> list:
        """Get the cutoff indices."""
        stop_position = len(df) - 1

        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        subseq_mid_idx = input_seq_len
        subseq_last_idx = input_seq_len + 1
        indices = []
        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
            subseq_first_idx += step_size
            subseq_mid_idx += step_size
            subseq_last_idx += step_size
        return indices


    def transform_ts_data_into_features_and_target(
        self,
        ts_data: pd.DataFrame,
        input_seq_len: int,
        step_size: int
    ) -> pd.DataFrame:
        """
        Slices and transposes data from time-series format into a (features, target)
        format that we can use to train Supervised ML models
        """
        assert set(ts_data.columns) == {'Date', 'Rides', 'StationID'}

        location_ids = ts_data['StationID'].unique()
        features = pd.DataFrame()
        targets = pd.DataFrame()
        
        for location_id in tqdm(location_ids):

            # keep only ts data for this `location_id`
            ts_data_one_location = ts_data.loc[
                ts_data.StationID == location_id, 
                ['Date', 'Rides']
            ]

            # pre-compute cutoff indices to split dataframe rows
            indices = self.get_cutoff_indices(
                ts_data_one_location,
                input_seq_len,
                step_size
            )

            # slice and transpose data into numpy arrays for features and targets
            n_examples = len(indices)
            x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
            y = np.ndarray(shape=(n_examples), dtype=np.float32)
            pickup_hours = []
            for i, idx in enumerate(indices):
                x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['Rides'].values
                y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['Rides'].values.item()
                # Ensure we preserve the datetime information
                pickup_hour = ts_data_one_location.iloc[idx[1]]['Date']
                # Convert to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(type(pickup_hour)):
                    pickup_hour = pd.to_datetime(pickup_hour)
                pickup_hours.append(pickup_hour)

            # numpy -> pandas
            features_one_location = pd.DataFrame(
                x,
                columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
            )
            features_one_location['pickup_hour'] = pickup_hours
            features_one_location['pickup_location_id'] = location_id

            # numpy -> pandas
            targets_one_location = pd.DataFrame(y, columns=[f'target_rides_next_hour'])

            # concatenate results
            features = pd.concat([features, features_one_location])
            targets = pd.concat([targets, targets_one_location])

        features.reset_index(inplace=True, drop=True)
        targets.reset_index(inplace=True, drop=True)

        # Let's return a single pandas dataframe with the features and targets
        df = pd.concat([features, targets], axis=1)

        return df

    def split_data(self, df: pd.DataFrame, cutoff_date: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into training and testing sets."""
        cutoff_date = pd.to_datetime(cutoff_date)
        train_df = df[df['pickup_hour'] < cutoff_date].reset_index(drop=True)
        test_df = df[df['pickup_hour'] >= cutoff_date].reset_index(drop=True)
        return train_df, test_df

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )