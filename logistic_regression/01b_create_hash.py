import time

from pyspark.sql import SQLContext
from pyspark.sql import SparkSession 

from pyspark.sql.types import *
from pyspark.ml.feature import  FeatureHasher

from pyspark.conf import SparkConf


app_name = "create_hash"
conf=SparkConf()
conf.set("spark.sql.parquet.compression.codec", "snappy")


spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

spark

for object in sc.getConf().getAll():
    print(object)

INPUT_FILE = 'gs://261_projectdata/261project_data/df.parquet'
OUT_FILE = 'gs://261_projectdata/261project_data/df_hash.parquet'
#parameter for setting the number of mapped feature space
NUM_FEATURES = 2**19

cat_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
           'n8', 'n9', 'n10', 'n11', 'n12',
           'cat13','cat14', 'cat15', 'cat16', 'cat17',
           'cat18', 'cat19', 'cat20', 'cat21', 'cat22',
           'cat23', 'cat24', 'cat25', 'cat26', 'cat27',
           'cat28', 'cat29', 'cat30', 'cat31', 'cat32',
           'cat33', 'cat34', 'cat35', 'cat36', 'cat37',
           'cat38', 'cat39']

df_pq = spark.read.load(INPUT_FILE)
#Create hash feature set
hasher = FeatureHasher(inputCols=cat_cols, outputCol="features", numFeatures = NUM_FEATURES)

start = time.time()
print(f'Creating hashed features.. {c}')
hash_transformed = hasher.transform(df_pq)
time_taken = time.time() - start
print(f"... completed job in {time_taken} seconds")

hash_transformed.show(n=2, truncate=False)
#save to file
print('writing file to parquet...')
final_df.write.parquet(OUT_FILE, compression='snappy', mode='overwrite')
print('Done.')