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
# --pyspark_file=decision_tree.py \
# --instance_type=n1-highmem-4
#

import math
from datetime import datetime

from pyspark.context import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import (RandomForestClassificationModel,
                                       RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (OneHotEncoderEstimator, StringIndexer,
                                VectorAssembler, VectorIndexer)
from pyspark.mllib.util import MLUtils
from pyspark.sql.functions import col, column, udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import DoubleType

# needed for autoscaling in dataproc
SparkContext.setSystemProperty('spark.dynamicAllocation.enabled', 'true')
SparkContext.setSystemProperty('spark.shuffle.service.enabled', 'true')


def log_loss(actual, predicted, epsilon=1e-15):
    acutal = int(actual)
    predicted = predicted[0] # class 0

    # clip value between [epsion, 1-epsilon]
    predicted = max(predicted, epsilon)
    predicted = min(predicted, 1-epsilon)

    if actual == 1:
        return -math.log(predicted)
    else:
        return -math.log(1 - predicted)


sc = SparkContext()
spark = SparkSession(sc)

spark.udf.register('log_loss', log_loss, DoubleType())
log_loss_udf = udf(log_loss, DoubleType())

df = spark.read.parquet(
    'gs://w261-f19-team1/ohe_10k').\
    select([col('y').alias('label'), 'features']).\
    where(col('label').isNotNull())

# split test/train datasets
_, test = df.randomSplit([0.7, 0.3], seed=0)

rfModel = RandomForestClassificationModel.load('gs://w261-f19-team1/rf')

rfPredictions = rfModel.transform(test).cache()
rfPredictions = rfPredictions.select('label', 'prediction', 'rawPrediction', 'probability', log_loss_udf(
    'label', 'probability').alias('log_loss')).cache()

rfPredictions.show(100, truncate=False)

print(rfPredictions.groupBy().mean('log_loss').collect())

evaluator = BinaryClassificationEvaluator()
print('Random Forest Test AUROC: ', evaluator.evaluate(
    rfPredictions, {evaluator.metricName: "areaUnderROC"}))

rfPredictions.write.parquet('gs://w261-f19-team1/predictions_rf2')
