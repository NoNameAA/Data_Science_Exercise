import sys
from pyspark.sql import SparkSession, functions, types, Row, DataFrame
import re
import math

spark = SparkSession.builder.appName('correlate logs').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        # TODO        
        return m.group(1), m.group(2)
    else:
        return None



def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """    
    # return (row[0] is not None) and (row[1] is not None)    
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # log_lines = log_lines.map(lambda row: line_to_row(row)).toDF(['x', 'y'])
    
    # print(log_lines.head())
    # print(log_lines.take(1)[0][1][1])
    log_lines = log_lines.map(lambda row: line_to_row(row)) \
                        .filter(not_none)
    return log_lines
    # print(log_lines.take(5))
    # print(log_lines.map(lambda row: line_to_row(row)).head())
    # return log_lines.map(lambda row: line_to_row(row)) \
    #     .filter('_1' == None)
    # TODO: return an RDD of Row() objects


def main(in_directory):

    schema = types.StructType([ # commented-out fields won't be read
        types.StructField('x', types.StringType(), False),
        types.StructField('y', types.StringType(), False),
    ])

    logs = spark.createDataFrame(create_row_rdd(in_directory), schema=schema)
    
    logs = logs.withColumn('y', logs['y'].cast(types.DoubleType()))

    logs = logs.cache()

    count_logs = logs.groupBy(logs.x).count()
    
    bytes_logs = logs.groupBy(logs.x).sum()
    
    joined_logs = count_logs.join(bytes_logs, count_logs.x == bytes_logs.x).drop(count_logs.x)
    
    joined_logs = joined_logs.withColumnRenamed('x', 'hostname')
    joined_logs = joined_logs.withColumnRenamed('count', 'x')
    joined_logs = joined_logs.withColumnRenamed('sum(y)', 'y')

    joined_logs = joined_logs.withColumn('xx', joined_logs.x ** 2)
    joined_logs = joined_logs.withColumn('yy', joined_logs.y ** 2)
    joined_logs = joined_logs.withColumn('xy', joined_logs.x * joined_logs.y)

    
    joined_logs = joined_logs.cache()
    
    n = joined_logs.count()
    sum_x = joined_logs.groupBy().sum('x').collect()[0][0]
    sum_y = joined_logs.groupBy().sum('y').collect()[0][0]
    sum_xx = joined_logs.groupBy().sum('xx').collect()[0][0]
    sum_yy = joined_logs.groupBy().sum('yy').collect()[0][0]
    sum_xy = joined_logs.groupBy().sum('xy').collect()[0][0]
    # print(n)
    # print(sum_x)
    # print(sum_y)
    # print(sum_xx)
    # print(sum_yy)
    # print(sum_xy)
    # print(joined_logs.groupBy().sum('x').collect()[0])
    r = (n * sum_xy - sum_x * sum_y) / \
        ( math.sqrt(n * sum_xx - sum_x ** 2) * math.sqrt(n * sum_yy - sum_y ** 2) )
    
    # TODO: calculate r.

    # r = 0 # TODO: it isn't zero.
    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
