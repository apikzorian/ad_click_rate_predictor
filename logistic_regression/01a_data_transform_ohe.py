# To run, exxecute in bash shell
# python submit_job_to_bigger_cluster.py --project_id=${PROJECT_ID} --zone=us-east1-b --cluster_name=samt --gcs_bucket=${BUCKET_NAME} --key_file=$HOME/MIDS/w261.json --create_new_cluster --pyspark_file=data_transform_ohe.py


import time


from pyspark.sql import SQLContext
from pyspark.sql import SparkSession 


from pyspark.sql.types import *
from pyspark.sql.functions import lit, when, col, approx_count_distinct
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer, OneHotEncoderEstimator, VectorAssembler, FeatureHasher

from pyspark.conf import SparkConf

app_name = "ohe"

#set session configuration
conf=SparkConf()
conf.set("spark.sql.parquet.compression.codec", "snappy")
#Create Spark session
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
        
sc = spark.sparkContext
sqlContext = SQLContext(sc)

spark
#print session configuration
for object in sc.getConf().getAll():
    print(object)

INPUT_FILE = 'gs://261_projectdata/261project_data/df.parquet'
OUT_FILE = 'gs://261_projectdata/261project_data/df_ohe_10k.parquet'
#this variable takes the minimum category occurence count as the threshold
THRESHOLD = 10_000

# Column names which will be transformed
my_cats = ['cat13','cat14', 'cat15', 'cat16', 'cat17',
           'cat18', 'cat19', 'cat20', 'cat21', 'cat22',
           'cat23', 'cat24', 'cat25', 'cat26', 'cat27',
           'cat28', 'cat29', 'cat30', 'cat31', 'cat32',
           'cat33', 'cat34', 'cat35', 'cat36', 'cat37',
           'cat38', 'cat39']

my_nums = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
           'n8', 'n9', 'n10', 'n11', 'n12']


#load file as dataframe
df_pq = spark.read.load(INPUT_FILE)
print(f' Total number of Rows = {df_pq.count()}')


def transform_str_col(df, cat_name):
    
    df_uniq_counts = df.groupBy(cat_name).count()
    
    #get values that occur above the threshold and broadcast it
    keep_vars = sc.broadcast(df_uniq_counts.filter(df_uniq_counts['count'] > THRESHOLD)\
                            .select(df_uniq_counts[cat_name])\
                            .rdd.flatMap(lambda x: x).collect())
    
    #broadcast the value to replace the low occurance values
    replace_val = sc.broadcast('stash_' + str(cat_name))


    
    #name the new column
    cat_t = str(cat_name) + '_t'
    
    df = df.withColumn(cat_t, when(col(cat_name).isin(keep_vars.value), col(cat_name)).otherwise(lit(replace_val.value)))
    df = df.drop(cat_name)
    return df



def transform_num_col(df, cat_name):
  #list of stash variables named so its easy to identify in coefficient analysis
    my_nums = {'n1':1_000_001, 'n2':1_000_002,
               'n3':1_000_003, 'n4':1_000_004,
               'n5':1_000_005, 'n6':1_000_006,
               'n7':1_000_007, 'n8':1_000_008, 
               'n9':1_000_009, 'n10':1_000_010,
               'n11':1_000_011, 'n12':1_000_012}
    
    df_uniq_counts = df.groupBy(cat_name).count()
    
    #get values that occur above the threshold and broadcast it
    keep_vars = sc.broadcast(df_uniq_counts.filter(df_uniq_counts['count'] > THRESHOLD)\
                            .select(df_uniq_counts[cat_name])\
                            .rdd.flatMap(lambda x: x).collect())
    
    #broadcast the stash variable placeholder value to replace the low occurance values
    replace_val = sc.broadcast(my_nums[cat_name])


    
    #name the new column
    cat_t = str(cat_name) + '_t'
    #if value is in the broadcasted high occurence list, then keep the value, other replace it with the placeholder
    df = df.withColumn(cat_t, when(col(cat_name).isin(keep_vars.value), col(cat_name)).otherwise(lit(replace_val.value)))
    df = df.drop(cat_name)
    return df


tot_time = 0
for c in my_nums:

    start = time.time()
    print(f'Transforming Numerical column.. {c}')
    df_pq = transform_num_col(df_pq, c)
    time_taken = time.time() - start
    print(f"... completed job in {time_taken} seconds")
    tot_time += time_taken
print(f'total time taken = {tot_time}')

print('\n')
print('----'* 20)
print('\n')


tot_time = 0
for c in my_cats:

    start = time.time()
    print(f'Transforming Categorical column.. {c}')
    df_pq = transform_str_col(df_pq, c)
    time_taken = time.time() - start
    print(f"... completed job in {time_taken} seconds")
    tot_time += time_taken
print(f'total time taken = {tot_time}')

df_pq.cache()




#specify columns to turn from string categorical variables to numerical categorical representation
cat_cols = ['n1_t', 'n2_t', 'n3_t', 'n4_t', 'n5_t', 'n6_t', 'n7_t',
           'n8_t', 'n9_t', 'n10_t', 'n11_t', 'n12_t',
           'cat13_t','cat14_t', 'cat15_t', 'cat16_t', 'cat17_t',
           'cat18_t', 'cat19_t', 'cat20_t', 'cat21_t', 'cat22_t',
           'cat23_t', 'cat24_t', 'cat25_t', 'cat26_t', 'cat27_t',
           'cat28_t', 'cat29_t', 'cat30_t', 'cat31_t', 'cat32_t',
           'cat33_t', 'cat34_t', 'cat35_t', 'cat36_t', 'cat37_t',
           'cat38_t', 'cat39_t']

cat_str_indx = ['n1_t_Indx', 'n2_t_Indx', 'n3_t_Indx', 'n4_t_Indx', 'n5_t_Indx', 'n6_t_Indx', 'n7_t_Indx',
               'n8_t_Indx', 'n9_t_Indx', 'n10_t_Indx', 'n11_t_Indx', 'n12_t_Indx',
               'cat13_t_Indx','cat14_t_Indx', 'cat15_t_Indx', 'cat16_t_Indx', 'cat17_t_Indx',
               'cat18_t_Indx', 'cat19_t_Indx', 'cat20_t_Indx', 'cat21_t_Indx', 'cat22_t_Indx',
               'cat23_t_Indx', 'cat24_t_Indx', 'cat25_t_Indx', 'cat26_t_Indx', 'cat27_t_Indx',
               'cat28_t_Indx', 'cat29_t_Indx', 'cat30_t_Indx', 'cat31_t_Indx', 'cat32_t_Indx',
               'cat33_t_Indx', 'cat34_t_Indx', 'cat35_t_Indx', 'cat36_t_Indx', 'cat37_t_Indx',
               'cat38_t_Indx', 'cat39_t_Indx']

cat_vecs = ['n1v', 'n2v', 'n3v', 'n4v', 'n5v', 'n6v', 'n7v',
           'n8v', 'n9v', 'n10v', 'n11v', 'n12v',
           'cat13v','cat14v', 'cat15v', 'cat16v', 'cat17v',
           'cat18v', 'cat19v', 'cat20v', 'cat21v', 'cat22v',
           'cat23v', 'cat24v', 'cat25v', 'cat26v', 'cat27v',
           'cat28v', 'cat29v', 'cat30v', 'cat31v', 'cat32v',
           'cat33v', 'cat34v', 'cat35v', 'cat36v', 'cat37v',
           'cat38v', 'cat39v']

#convert categorical variables to numerical categories
indexers = [StringIndexer(inputCol= c, 
                         outputCol="{0}_Indx".format(c), 
                         handleInvalid="keep") 
           for c in cat_cols]
#Create a list representation of the numerical categorical representation
encoder = OneHotEncoderEstimator(inputCols = [indexer.getOutputCol() for indexer in indexers], 
                                  outputCols = cat_vecs, 
                                  dropLast = True)
#adding a assembler step which creates a sparse representation of an indexed column
assembler = VectorAssembler(inputCols=encoder.getOutputCols(), 
                           outputCol = 'features')
#creating the full transform pipeline  
start = time.time()
print(f'Running pipeline to create sparse vectors.. ')                            
pipeline = Pipeline(stages = indexers + [encoder] + [assembler] )

model = pipeline.fit(df_pq)

transformed = model.transform(df_pq)

drop_cols = cat_str_indx + cat_vecs
final_df = transformed.drop(*drop_cols).cache()

time_taken = time.time() - start
print(f"... completed job in {time_taken} seconds")


print('writing file to parquet')
final_df.write.parquet(OUT_FILE, compression='snappy', mode='overwrite')
print('Done.')

