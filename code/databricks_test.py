from pyspark.sql import SparkSession

try:
    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.serverless(True).getOrCreate()
except ImportError:
    spark = SparkSession.builder.getOrCreate()

# Create example DataFrame
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
columns = ["name", "age"]
df = spark.createDataFrame(data, columns)

# Save DataFrame to a Databricks table
table_name = "dev.default.test_table"
df.write.mode("overwrite").saveAsTable(table_name)

# Read the table back
df_read = spark.table(table_name)
df_read.show()
