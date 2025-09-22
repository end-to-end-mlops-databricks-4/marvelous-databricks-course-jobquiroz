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
        columns = list(self.config.original_columns.values())
        # Read table, select columns and filter by date
        df = (self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.{self.table_name}")
                    .select(columns)
                    .filter(F.col('Fecha_Retiro').between(min_date, max_date))
                    )
        return df


    def preprocess_data(self, df: SparkDataFrame) -> SparkDataFrame:
        """Preprocess the data."""
        # Rename the columns using the mapping from config
        for new_name, original_name in self.config.original_columns.items():
            df = df.withColumnRenamed(original_name, new_name)

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

        # Convert temporary columns to strings
        for temp_col in self.config.temporary_columns:
            df = df.withColumn(temp_col, F.col(temp_col).cast("string"))

        # Concatenate them, but for Hour we only want the time part (e.g. 12:20:10), only the last 8 characters
        # Assuming Date is first and Hour is second in temporary_columns
        date_col, hour_col = self.config.temporary_columns[0], self.config.temporary_columns[1]
        pickup_datetime_col = self.config.datetime_features[0] # This is the new column name (PickUpDateTime)
        df = df.withColumn(pickup_datetime_col, F.concat(F.col(date_col), F.substring(F.col(hour_col), -9, 9)))

        # Convert PickUpDateTime to datetime
        df = df.withColumn(pickup_datetime_col, F.to_timestamp(pickup_datetime_col))

        # Floor the datetime to the nearest hour
        df = df.withColumn(pickup_datetime_col, F.date_trunc("hour", F.col(pickup_datetime_col)))

        # Keep only the columns we need
        station_id_col = self.config.cat_features[0]
        df = df.select(station_id_col, pickup_datetime_col)

        return df

    def get_agg_data(self, df: SparkDataFrame) -> pd.DataFrame:
        """Get the aggregated data."""
        # Count the number of rides
        station_id_col = self.config.cat_features[0]
        pickup_datetime_col = self.config.datetime_features[0]
        df = df.groupby([station_id_col, pickup_datetime_col]).count().withColumnRenamed("count", "Rides")

        # Sort by Rides descending
        df = df.sort(F.desc("Rides"))

        # Convert to pandas
        df = df.toPandas()

        # Add missing slots
        df = self.add_missing_slots(df)

        return df

    def add_missing_slots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the missing slots of time."""
        station_id_col = self.config.cat_features[0]
        pickup_datetime_col = self.config.datetime_features[0]
        location_ids = df[station_id_col].unique()
        full_range = pd.date_range(df[pickup_datetime_col].min(), df[pickup_datetime_col].max(), freq='h')
        output = pd.DataFrame()
        for location_id in tqdm(location_ids):
            location_df = df[df[station_id_col] == location_id]
            location_df = location_df.set_index(pickup_datetime_col)
            location_df.index = pd.DatetimeIndex(location_df.index)
            location_df = location_df.reindex(full_range, fill_value=0)
            location_df[station_id_col] = location_id
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
        station_id_col = self.config.cat_features[0]
        expected_columns = {'Date', 'Rides', station_id_col}
        assert set(ts_data.columns) == expected_columns

        location_ids = ts_data[station_id_col].unique()
        features = pd.DataFrame()
        targets = pd.DataFrame()
        
        for location_id in tqdm(location_ids):

            # keep only ts data for this `location_id`
            ts_data_one_location = ts_data.loc[
                ts_data[station_id_col] == location_id, 
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
            features_one_location[f'pickup_{station_id_col.lower()}_id'] = location_id

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