# python submit_job_to_bigger_cluster.py --project_id=${PROJECT_ID} --zone=us-east1-b --cluster_name=samt --gcs_bucket=${BUCKET_NAME} --key_file=$HOME/MIDS/w261.json --create_new_cluster --pyspark_file=lr_model.py

import time
import math


from pyspark.sql import SQLContext
from pyspark.sql import SparkSession 


from pyspark.sql.types import *
from pyspark.sql.functions import lit, when, col, approx_count_distinct, mean, log, udf

from pyspark.ml.classification import LogisticRegression

from pyspark.conf import SparkConf


app_name = "logistic_regression"

#set spark session options
conf=SparkConf()
conf.set("spark.sql.parquet.compression.codec", "snappy")
#initiate the spark session
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

spark
#print all the session options and settings
for object in sc.getConf().getAll():
    print(object)

#set the global variables
MAX_ITER = 10       #number of model training epochs
THRESHOLD = 0.45    #model training threshold for classification probability
EPSILON = 1e-16     #variable to force bounded solution for log trransforms

#import one hot encoded file
INPUT_FILE = 'gs://261_projectdata/261project_data/df_ohe_10k.parquet'
OUTPUT_FILE = 'gs://261_projectdata/results/ohe_10k_result.csv'

df_pq = spark.read.parquet(INPUT_FILE)\
                .select('y', 'features')

print(f' Total number of Rows = {df_pq.count()}')
#remove records where the label is missing
df_pq = df_pq.na.drop(subset = ["y"])

print(f' Dropped NULL values from prediction variable. Total number of Rows = {df_pq.count()}')
#create train-test split
train_y_all, test_y_all = df_pq.randomSplit([0.8, 0.2])
df_pq.unpersist()

#change the name of the dependent variable column
test = test_y_all.selectExpr("y as labels", "features")
print(f' Total number of Rows in the TEST set = {test.count()}')
test.show(n=5)
#force a binary classification 0 or 1 on the training data and save as column 'labels'
train_y = train_y_all.withColumn('labels', when(col('y') > 0.0, lit(1)).otherwise(lit(0)))
train_y_all.unpersist()
#remove the old dependent variable
train = train_y.drop('y')
train_y.unpersist()

print(f' Total number of Rows in the TRAIN set = {train.count()}')

train.show(n=10)


#Get the percentage of values in the positive class, to set the baseline log loss
mean_prob = train.select(mean(col('labels'))).collect()[0]['avg(labels)']
print(f'Positive class in the training data = {mean_prob * 100} %')

#assign mean probability as the baseline probability estimate
train_prob = train.withColumn('base_prob', lit(mean_prob))
train.unpersist()

# calculate baseilne logloss  
train_ll = train_prob.withColumn('logloss', when(col('labels') > 0.0, - log(col('base_prob') + EPSILON))\
                                       .otherwise( - log(1.0 - col('base_prob') + EPSILON)))
train_prob.unpersist()
#get the mean of the logloss
mean_ll = train_ll.select(mean(col('logloss'))).collect()[0]['avg(logloss)']
print(f'Baseline Log loss for the training data = {mean_ll}')



#specify the model and hyperparameters
lr = LogisticRegression(featuresCol = 'features',
                        labelCol='labels', 
                        maxIter = MAX_ITER, 
                        standardization = False,
                        elasticNetParam = 0.0, 
                        threshold = THRESHOLD
                        )

start = time.time()
print('Training model...')
lr_model = lr.fit(train_ll)
time_taken = time.time() - start
print(f'Training completed ...in {time_taken}')




def sigmoid(rawPrediction):
    #apply affine transformation
    return 1 / (1 + math.exp(- rawPrediction))

def extract_from_vector(vec, i):
    """ Input: VectorUDT data type
        Output: float type at index (i)
    """
    try:
        return float(vec[i])
    except ValueError:
        return None

#register UDFs    
get_probability = udf(sigmoid, DoubleType())
get_val = udf(extract_from_vector, DoubleType())

#Get predictions on test data
result = lr_model.transform(test)

#Apply affine transformation to the linear regression output
result2 = result.withColumn('calc_prob', get_probability(get_val("rawPrediction", lit(1))))
# calculate log loss
result3 = result2.withColumn('logloss', when(col('labels') > 0.0, - log(col('calc_prob') + EPSILON))\
                                       .otherwise( - log(1.0 - col('calc_prob') + EPSILON)))


test_mean_ll = result3.select(mean(col('logloss'))).collect()[0]['avg(logloss)']
print(f'Baseline Log loss for the training data = {mean_ll}')
print(f'MODEL Log loss for the training data = {test_mean_ll}')

result3.select(['features', 'labels' ,'prediction', 'calc_prob', 'probability', 'logloss']).show(n=10, truncate=True)

analyze = result3.select('labels', 'prediction')
analyze = analyze.withColumn("prediction", analyze["prediction"].cast(IntegerType()))
analyze = analyze.withColumn("labels", analyze["labels"].cast(IntegerType()))
result3.unpersist()
#Create SQL queries table
sqlContext.registerDataFrameAsTable(analyze, "results")
#Get metrics for precision recall calculations
TP = spark.sql("""SELECT COUNT(prediction)
                    FROM results
                    WHERE labels = 1 AND prediction = 1
                    """).collect()[0][0]

TN = spark.sql("""SELECT COUNT(prediction)
                    FROM results
                    WHERE labels = 0 AND prediction = 0
                    """).collect()[0][0]

FP = spark.sql("""SELECT COUNT(prediction)
                    FROM results
                    WHERE labels = 0 AND prediction = 1
                    """).collect()[0][0]

FN = spark.sql("""SELECT COUNT(prediction)
                    FROM results
                    WHERE labels = 1 AND prediction = 0
                    """).collect()[0][0]


print(f'Precision = {TP/(TP + FP) * 100} %')

print(f'Recall = {TP/(TP + FN) * 100} %')

features = [x["name"] for x in sorted(train_ll.schema["features"].metadata["ml_attr"]["attrs"]["binary"], 
                                      key=lambda x: x["idx"]
                                     )]

#Extract feature names and indices from table metadata
schema = StructType([StructField("feature", StringType()),
                    StructField("coeff", FloatType())
                    ])

# Create dataframe of coefficients
result_df = spark.createDataFrame(zip(features, lr_model.coefficients.tolist()), schema=schema)
result_df.show(n=5)

#save to CSV file
result_df.coalesce(1).write.csv(OUTPUT_FILE, mode='overwrite')