# to create autoscaling policy:
# gcloud dataproc autoscaling-policies import autoscaling-policy --source=autoscaling.yml
#
# to run:
# python dataproc/submit_job_to_cluster.py \
# --project_id=${PROJECT_ID} \
# --zone=${ZONE} \
# --cluster_name=${CLUSTER_NAME} \
# --gcs_bucket=${BUCKET} \
# --key_file=${KEY} \
# --create_new_cluster \
# --pyspark_file=train.py \
# --instance_type=n1-highmem-4

import math
import random
from datetime import datetime

import numpy as np
from pyspark.context import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import (DecisionTreeClassifier,
                                       RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (OneHotEncoderEstimator, StringIndexer,
                                VectorAssembler, VectorIndexer)
from pyspark.mllib.util import MLUtils
from pyspark.sql import Row
from pyspark.sql.functions import col, column, udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import DoubleType
from sklearn import neighbors

# needed for autoscaling in dataproc
SparkContext.setSystemProperty('spark.dynamicAllocation.enabled', 'true')
SparkContext.setSystemProperty('spark.shuffle.service.enabled', 'true')

sc = SparkContext()
spark = SparkSession(sc)

def oversample(df, target_col='label', feature_col='features', majority_class='0.0', minority_class='1.0', percentage_over=2):
    minority = df[df[target_col] == minority_class].cache()

    for i in range(percentage_over):
        df = df.unionAll(minority)

    return df

def log_loss(actual, predicted, epsilon=1e-15):
    acutal = int(actual)
    predicted = predicted[0]  # class 0

    # clip value between [epsion, 1-epsilon]
    predicted = max(predicted, epsilon)
    predicted = min(predicted, 1-epsilon)

    if actual == 1:
        return -math.log(predicted)
    else:
        return -math.log(1 - predicted)


spark.udf.register('log_loss', log_loss, DoubleType())
log_loss_udf = udf(log_loss, DoubleType())

df = spark.read.parquet(
    'gs://w261-f19-team1/ohe_10k_numeric').\
    select(['label', 'features']).\
    where(col('label').isNotNull())

# split test/train datasets
train, test = df.randomSplit([0.7, 0.3], seed=0)

train = train.cache()
test = test.cache()

print('original dataset:')
print('train count = ', train.count())
print('train class = 0', train.filter(train.label == '0.0').count())
print('train class = 1', train.filter(train.label == '1.0').count())
print('test count = ', test.count())

print('-' * 100)

# make 2 artificial examples of clicks per each example since we have a 25/75 split of click/non-click examples
train = oversample(train, minority_class = '1.0', majority_class = '0.0', percentage_over = 2).cache()

print('oversample dataset:')
print('train count = ', train.count())
print('train class = 0', train.filter(train.label == '0.0').count())
print('train class = 1', train.filter(train.label == '1.0').count())

# random forest training
n_trees = 40
max_depth = 5
max_bins = 32

print(f'Random forest classifier with {n_trees} trees, max depth of {max_depth} and max bins of {max_bins}')

rf = RandomForestClassifier(featuresCol='features',
                            labelCol='label', numTrees=n_trees, maxDepth=max_depth, maxBins=max_bins)
rfModel = rf.fit(train)
rfPredictions = rfModel.transform(test)
rfPredictions = rfPredictions.select('label',
                                     'prediction',
                                     'rawPrediction',
                                     'probability',
                                     log_loss_udf('label', 'probability').alias('log_loss')).\
    cache()

print('Log loss: ', rfPredictions.groupBy().mean('log_loss').collect()[0]['avg(log_loss)'])

rfPredictions.filter(rfPredictions.label == '0.0').show(50, truncate=False)
rfPredictions.filter(rfPredictions.label == '1.0').show(50, truncate=False)

rfModel.save(
    f'gs://w261-f19-team1/oversample_rf_{n_trees}_{max_depth}_{max_bins}/model')
rfPredictions.write.parquet(
    f'gs://w261-f19-team1/oversample_rf_{n_trees}_{max_depth}_{max_bins}/predictions')
