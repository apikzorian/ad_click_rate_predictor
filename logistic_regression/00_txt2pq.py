import time

from pyspark.sql import SQLContext
from pyspark.sql import SparkSession 
from pyspark.sql.types import *

from pyspark.conf import SparkConf


app_name = "final_project"

#Set Spark session configuration
conf=SparkConf()
conf.set("spark.sql.parquet.compression.codec", "snappy")
#Create Spark session
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .master(master)\
        .getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

spark
#print session settings
for object in sc.getConf().getAll():
    print(object)


INPUT_FILES = 'gs://261_projectdata/261project_data/train/train{00,01,02,03,04,05,06,07,08,09,10}'
OUT_FILES = 'gs://261_projectdata/261project_data/df.parquet'


#specify schema on read
schema = StructType([
    StructField('y', IntegerType()),
    StructField('n1', IntegerType()),
    StructField('n2', IntegerType()),
    StructField('n3', IntegerType()),
    StructField('n4', IntegerType()),
    StructField('n5', LongType()),
    StructField('n6', IntegerType()),
    StructField('n7', IntegerType()),
    StructField('n8', IntegerType()),
    StructField('n9', IntegerType()),
    StructField('n10', IntegerType()),
    StructField('n11', IntegerType()),
    StructField('n12', IntegerType()),
    StructField('cat13', StringType()),
    StructField('cat14', StringType()),
    StructField('cat15', StringType()),
    StructField('cat16', StringType()),
    StructField('cat17', StringType()),
    StructField('cat18', StringType()),
    StructField('cat19', StringType()),
    StructField('cat20', StringType()),
    StructField('cat21', StringType()),
    StructField('cat22', StringType()),
    StructField('cat23', StringType()),
    StructField('cat24', StringType()),
    StructField('cat25', StringType()),
    StructField('cat26', StringType()),
    StructField('cat27', StringType()),
    StructField('cat28', StringType()),
    StructField('cat29', StringType()),
    StructField('cat30', StringType()),
    StructField('cat31', StringType()),
    StructField('cat32', StringType()),
    StructField('cat33', StringType()),
    StructField('cat34', StringType()),
    StructField('cat35', StringType()),
    StructField('cat36', StringType()),
    StructField('cat37', StringType()),
    StructField('cat38', StringType()),
    StructField('cat39', StringType()),
])

start = time.time()
print('Creating dataframe..')
df = spark.read.load(INPUT_FILES, format='csv', sep='\t', header='false', schema=schema)
print(f"... completed job in {time.time() - start} seconds")


#Take a peek at the dataframe
print(df.select('y','n1','cat39').show(n=3))
#get count
print(df.count())

#write DF to parquet file
start = time.time()
print('Writing dataframe to parquet format..')
df.write.parquet(OUT_FILES, compression='snappy', mode='overwrite')
print(f"... completed job in {time.time() - start} seconds")

