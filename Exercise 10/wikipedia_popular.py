import sys
import re
from pyspark.sql import SparkSession, functions, types


spark = SparkSession.builder.appName('reddit averages').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

def datetime_filter(s):
	# file:/home/sitel/Desktop/353/exercise10/pagecounts-1/pagecounts-20160801-130000.gz
	r = s.find('.') - 4
	l = r - 11
	return s[l:r]

def main(in_directory, out_directory):
	df = spark.read.csv(in_directory, sep = ' ').withColumn('filename', functions.input_file_name())
	
	df = df.withColumnRenamed('_c0', 'language')
	df = df.withColumnRenamed('_c1', 'title')
	df = df.withColumnRenamed('_c2', 'view')
	df = df.withColumnRenamed('_c3', 'bytes')

	udf_datetime_filter = functions.udf(datetime_filter, returnType = types.StringType())
	df = df.withColumn('datetime', udf_datetime_filter(df['filename']))
	df = df.withColumn('view', df['view'].cast(types.IntegerType()))
	

	df = df.filter( (df['language'].startswith('en')) & (df['title'] != 'Main_Page') & (df['title'].startswith('Special:') == False) )
	
	df = df.cache()

	max_data = df.groupBy('datetime').max('view')
	max_data = max_data.withColumnRenamed('max(view)', 'view')
	

	joined_data = df.join(max_data, (df.datetime == max_data.datetime) & (df.view == max_data.view), 'inner' ).drop(df.view).drop(df.datetime)
	joined_data = joined_data.drop('language', 'bytes', 'filename')
	joined_data = joined_data.sort('datetime')
	joined_data = joined_data.select('datetime', 'title', 'view')

	# joined_data.show()

	joined_data.write.csv(out_directory, mode='overwrite')
	


if __name__ == "__main__":
	in_directory = sys.argv[1]
	out_directory = sys.argv[2]
	main(in_directory, out_directory)