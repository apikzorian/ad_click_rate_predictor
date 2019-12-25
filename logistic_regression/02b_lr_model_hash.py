


import time
import math

from pyspark.sql import SQLContext
from pyspark.sql import SparkSession 

from pyspark.conf import SparkConf

from pyspark.sql.types import *
from pyspark.sql.functions import lit, when, col, approx_count_distinct, mean, log, udf
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import FeatureHasher



app_name = "logistic_regression"

conf=SparkConf()

spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

spark

for object in sc.getConf().getAll():
    print(object)
#set the global variables
NUM_FEATURES = 2**18  #parameter for setting the number of mapped feature space
MAX_ITER = 10       #number of model training epochs
THRESHOLD = 0.45    #model training threshold for classification probability
EPSILON = 1e-16     #variable to force bounded solution for log trransforms


INPUT_FILE = 'gs://261_projectdata/261project_data/df.parquet'



my_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
           'n8', 'n9', 'n10', 'n11', 'n12',
           'cat13','cat14', 'cat15', 'cat16', 'cat17',
           'cat18', 'cat19', 'cat20', 'cat21', 'cat22',
           'cat23', 'cat24', 'cat25', 'cat26', 'cat27',
           'cat28', 'cat29', 'cat30', 'cat31', 'cat32',
           'cat33', 'cat34', 'cat35', 'cat36', 'cat37',
           'cat38', 'cat39']



df_pq = spark.read.parquet(INPUT_FILE)


print(f'Creating Hashed feature with {NUM_FEATURES} number of vectors ...')
start = time.time()

hasher = FeatureHasher(inputCols=my_cols, outputCol="features", numFeatures = NUM_FEATURES)

hash_transformed = hasher.transform(df_pq).select('y', 'features')
print(f'Hashed feature vector completed ...in {time.time() - start}')

df_pq.unpersist()

hash_transformed.show(n=5, truncate=True)
#create train-test split
print('Creating train-test split..')
train_y_all, test_y_all = hash_transformed.randomSplit([0.8, 0.2], seed=10)
hash_transformed.unpersist()
#change the name of the dependent variable column
print('Creating test set..')
test = test_y_all.selectExpr("y as labels", "features")
test.show(n=5)
#force a binary classification 0 or 1 on the training data and save as column 'labels'
print('Creating train set..')
train_y = train_y_all.withColumn('labels', when(col('y') > 0.0, lit(1)).otherwise(lit(0)))
train_y_all.unpersist()

train = train_y.drop('y')
train_y.unpersist()

train.show(n=10)


#Get the percentage of values in the positive class, to set the baseline log loss
mean_prob = train.select(mean(col('labels'))).collect()[0]['avg(labels)']
print(f'Positive class in the training data = {mean_prob * 100} %')

#assign mean probability as the baseline probability estimate
train_prob = train.withColumn('base_prob', lit(mean_prob))
train.unpersist()

# calculate baseilne logloss  
train_ll = train_prob.withColumn('logloss', when(col('labels') == 1.0, - log(col('base_prob') + EPSILON))\
                                       .otherwise( - log(1.0 - col('base_prob') + EPSILON)))
train_prob.unpersist()

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

analyze = result3.select('labels', 'prediction')
analyze = analyze.withColumn("prediction", analyze["prediction"].cast(IntegerType()))
analyze = analyze.withColumn("labels", analyze["labels"].cast(IntegerType()))

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

P = spark.sql("""SELECT COUNT(labels)
                    FROM results
                    WHERE labels = 1
                    """).collect()[0][0]

N = spark.sql("""SELECT COUNT(labels)
                    FROM results
                    WHERE labels = 0
                    """).collect()[0][0]

print(f'Precision = {TP/(TP + FP) * 100} %')

print(f'Recall = {TP/(TP + FN) * 100} %')