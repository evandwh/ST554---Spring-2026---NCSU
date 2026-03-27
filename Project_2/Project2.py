############################################################################
#
# Author - Evan Whitfield
# Last Editted - 3/26/26
# Purpose - ST554 Project 2
#         - Creates a class that 
#############################################################################

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, isnan
from functools import reduce
from pyspark.sql.types import *
import pandas as pd

class SparkDataChecker:
    def __init__(self, dataframe):
        self.df = dataframe

    #reading in data into a SparkDataChecker class
    @classmethod
    def read_csv(cls, spark_session, file_path):
        df = spark_session.read.load(file_path, format="csv", header=True, inferSchema=True)
        return cls(df)

    @classmethod
    def from_pandas(cls, spark_session, pandas_df):
        df = spark_session.createDataFrame(pandas_df)
        return cls(df)

    #checks to see if the given column is in the data set
    def column_exists(self, column_name):
        dtype_dict = dict(self.df.dtypes)

        if column_name not in dtype_dict:
            print(f"Column '{column_name}' does not exist.")
            return False

        return True

    #checks to see if the given column is a string type
    def _is_string_column(self, column_name):
        if not self.column_exists(column_name):
            return False
        
        dtype_dict = dict(self.df.dtypes)

        return dtype_dict[column_name] == 'string'

    #checks to see if the given column belongs to a list of numeric types
    def _is_numeric_column(self, column_name):
        numeric_types = ['int', 'bigint', 'double', 'float', 'long', 'decimal', 'longint', 'bigint']

        if not self.column_exists(column_name):
            return False

        dtype_dict = dict(self.df.dtypes)

        return dtype_dict[column_name] in numeric_types
    
    #checks to see which values in given column are within the range provided. Must be numeric, and must define at least part of range.
    def check_in_range(self, column_name, lower = None, upper = None):
        
        #checking conditions
        if lower is None and upper is None:
            print("Must give at least one of lower or upper.")
            return self

        if not self._is_numeric_column(column_name):
            print(f"Column '{column_name}' is not numeric.")
            return self

        #using infinity as bounds if no bound is provided
        if lower is None:
            lower = float("-inf")

        if upper is None:
            upper = float("inf")

        #returning list of boolean values if column value fits within range
        tf_list = self.df[column_name].between(lower, upper)

        #returning Null when given column is Null
        tf_list = when(col(column_name).isNull() | isnan(col(column_name)),
            None).otherwise(tf_list)

        #adding boolean list to self
        self = self.df.withColumn(f"{column_name}_in_range", tf_list)
        
        return self
    
    #checking to see string is within a given list of values
    def str_isin(self, column_name, values):
        if not self._is_string_column(column_name):
            print(f"Column '{column_name}' is not a string.")
            return self

        tf_column = self.df[column_name].isin(values)
        self = self.df.withColumn(f"{column_name}_is_in", tf_column)
        return self

    #checks to see if values in given column is Null
    def NullCheck(self, column_name):
        self = self.df.withColumn(f"{column_name}_is_null", col(column_name).isNull())
        return self

    #summarize with the min and max of all numeric columns if none provided, or of given column
    def summarize_min_max(self, column_name=None, groupedby=None):
        #Case 1 - Column is provided
        if column_name:
            #checking to see if column is numeric
            if not self._is_numeric_column(column_name):
                print(f"'{column_name}’ is not numeric.")
                return None
        
            #checking to see if groupedby has been used
            if groupedby:
                result = (self.df.groupBy(groupedby).agg(
                    F.min(column_name).alias("min"),
                    F.max(column_name).alias("max")
                    )
                )
            #returning min max if not grouped
            else:
                result = self.df.agg(
                    F.min(column_name).alias("min"),
                    F.max(column_name).alias("max")
                )
            return result.toPandas()
        
        #Case 2 - no column has been provided
        #Need to determine the numeric columns in the data set
        numeric_cols = [c for c in self.df.columns if self._is_numeric_column(c)]
        if len(numeric_cols) == 0:
            return None
        
        minmaxes = []
        for col in numeric_cols:
            #case 2a - GroupedBy has been used
            if groupedby:
                minmax = (self.df.groupBy(groupedby).agg(
                    F.min(F.col(f"`{col}`")).alias(f"{col}_min"),
                    F.max(F.col(f"`{col}`")).alias(f"{col}_max")
                    )
                )
                
            #case 2b - GroupedBy has NOT been used
            else:
                minmax = self.df.agg(
                    F.min(F.col(f"`{col}`")).alias(f"{col}_min"),
                    F.max(F.col(f"`{col}`")).alias(f"{col}_max")
                )
                
            #append all of the min max results together
            minmaxes.append(minmax)

        # Merge results
        if groupedby:
            result = reduce(lambda left, right: left.join(right, on=groupedby), minmaxes)
        
        else:
            result = reduce(lambda left, right: left.crossJoin(right), minmaxes)
        
        return result.toPandas()
        
    #Return how counts of different values in given string columns. Fails if no string column given.
    def summarize_counts(self, col1, col2=None):
        
        #checking to see if columns are string columns
        check1 = self._is_string_column(col1)
        check2 = False
        if col2:
            check2 = self._is_string_column(col2)
            
        # Printing results of string column check
        if not check1:
            print(f"'{col1}' is not a string column.")
        if col2 and not check2:
            print(f"'{col2}' is not a string column.")
        
        #if both columns are NOT strings
        if not check1 and not check2:
            return None
        
        #Col2 is a string, but Col1 is not a string
        if col2 and not check1:
            result = self.df.groupBy(col2).count()
            
        #Both Col1 and Col2 are strings
        elif check1 and check2:
            result = self.df.groupBy(col1, col2).count()
        
        #Col1 is a string, Col2 either not a string or not provided
        else:
            result = self.df.groupBy(col1).count()
        
        return result.toPandas()