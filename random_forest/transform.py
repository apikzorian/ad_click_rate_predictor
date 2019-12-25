# To run, execute in bash shell
# python submit_job_to_cluster.py --project_id=${PROJECT_ID} --zone=${ZONE} --cluster_name=${CLUSTER_NAME} --gcs_bucket=${BUCKET} --key_file=${KEY} --create_new_cluster --pyspark_file=transform.py

import time

from pyspark.conf import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import (FeatureHasher, IndexToString,
                                OneHotEncoderEstimator, StringIndexer,
                                VectorAssembler)
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import approx_count_distinct, col, lit, when
from pyspark.sql.types import *

app_name = "ohe"


conf = SparkConf()
conf.set("spark.sql.parquet.compression.codec", "snappy")

spark = SparkSession\
    .builder\
    .appName(app_name)\
    .getOrCreate()

sc = spark.sparkContext
sqlContext = SQLContext(sc)

print('Spark version: ', sc.version)

for object in sc.getConf().getAll():
    print(object)

INPUT_FILE = 'gs://w261-f19-team1/train'
OUT_FILE = 'gs://w261-f19-team1/ohe_10k_numeric'
THRESHOLD = 10_000

target_col = '_c0'
numeric_columns = [f'_c{i}' for i in range(1, 13)]
category_columns = [f'_c{i}' for i in range(13, 30)]

df_pq = spark.read.load(INPUT_FILE)

# StringIndexer requires string not boolean type
df_pq = df_pq.withColumn(target_col, col(target_col).cast('string'))

print(f' Total number of Rows = {df_pq.count()}')


def transform_str_col(df, cat_name):

    df_uniq_counts = df.groupBy(cat_name).count()

    # get values that occur above the threshold and broadcast it
    keep_vars = sc.broadcast(df_uniq_counts.filter(df_uniq_counts['count'] > THRESHOLD)
                             .select(df_uniq_counts[cat_name])
                             .rdd.flatMap(lambda x: x).collect())

    # broadcast the value to replace the low occurance values
    replace_val = sc.broadcast('stash_' + str(cat_name))

    # name the new column
    cat_t = str(cat_name) + '_t'

    df = df.withColumn(cat_t, when(col(cat_name).isin(
        keep_vars.value), col(cat_name)).otherwise(lit(replace_val.value)))
    df = df.drop(cat_name)
    return df


tot_time = 0
for c in category_columns:

    start = time.time()
    print(f'Transforming Categorical column.. {c}')
    df_pq = transform_str_col(df_pq, c)
    time_taken = time.time() - start
    print(f"... completed job in {time_taken} seconds")
    tot_time += time_taken
print(f'total time taken = {tot_time}')

df_pq.cache()

# transformed category column names
cat_cols = [f'{col}_t' for col in category_columns]

# indexed category column names
cat_str_indx = [f'{col}_Indx' for col in cat_cols]

# vectorized category column names
cat_vecs = [f'{col}v' for col in category_columns]

indexers = [StringIndexer(inputCol=c,
                          outputCol="{0}_Indx".format(c),
                          handleInvalid="keep")
            for c in cat_cols]

encoder = OneHotEncoderEstimator(inputCols=[indexer.getOutputCol() for indexer in indexers],
                                 outputCols=cat_vecs,
                                 dropLast=True)

assembler = VectorAssembler(inputCols=numeric_columns + encoder.getOutputCols(),
                            outputCol='features',
                            handleInvalid='keep')

label_indexer = StringIndexer(inputCol=target_col, outputCol='label')

start = time.time()
print(f'Running pipeline to create sparse vectors.. ')
pipeline = Pipeline(
    stages=indexers + [encoder] + [assembler] + [label_indexer])

model = pipeline.fit(df_pq)

transformed = model.transform(df_pq)

drop_cols = cat_str_indx + cat_vecs
final_df = transformed.drop(*drop_cols).cache()

time_taken = time.time() - start
print(f"... completed job in {time_taken} seconds")


print('writing file to parquet')
final_df.write.parquet(OUT_FILE, compression='snappy', mode='overwrite')
print('Done.')
