import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

schema = types.StructType([ # commented-out fields won't be read
    #types.StructField('archived', types.BooleanType(), False),
    types.StructField('author', types.StringType(), False),
    #types.StructField('author_flair_css_class', types.StringType(), False),
    #types.StructField('author_flair_text', types.StringType(), False),
    #types.StructField('body', types.StringType(), False),
    #types.StructField('controversiality', types.LongType(), False),
    #types.StructField('created_utc', types.StringType(), False),
    #types.StructField('distinguished', types.StringType(), False),
    #types.StructField('downs', types.LongType(), False),
    #types.StructField('edited', types.StringType(), False),
    #types.StructField('gilded', types.LongType(), False),
    #types.StructField('id', types.StringType(), False),
    #types.StructField('link_id', types.StringType(), False),
    #types.StructField('name', types.StringType(), False),
    #types.StructField('parent_id', types.StringType(), True),
    #types.StructField('retrieved_on', types.LongType(), False),
    types.StructField('score', types.LongType(), False),
    #types.StructField('score_hidden', types.BooleanType(), False),
    types.StructField('subreddit', types.StringType(), False),
    #types.StructField('subreddit_id', types.StringType(), False),
    #types.StructField('ups', types.LongType(), False),
])


def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=schema)

    comments.cache()
    average = comments.groupBy('subreddit').avg('score')
    average = average.withColumnRenamed('avg(score)', 'avg_score')
    average = average.filter( average['avg_score'] > 0 )
    # average = functions.broadcast(average)
    # average.show()
    joined_comments = comments.join(average, average.subreddit == comments.subreddit ).drop(comments.subreddit)
    joined_comments = joined_comments.withColumn('relative_score', joined_comments['score'] / joined_comments['avg_score'] )
    joined_comments.cache()

    max_score_comments = joined_comments.groupBy('subreddit').max('relative_score')
    max_score_comments = max_score_comments.withColumnRenamed('max(relative_score)', 'max_relative_score')
    # max_score_comments = functions.broadcast(max_score_comments)

    result_table = joined_comments.join(max_score_comments, (joined_comments.subreddit == max_score_comments.subreddit) & (joined_comments.relative_score == max_score_comments.max_relative_score) ).drop(joined_comments.subreddit)
    result_table = result_table.drop('avg_score', 'score', 'max_relative_score')

    result_table = result_table.select('subreddit', 'author', 'relative_score')
    result_table = result_table.withColumnRenamed('relative_score', 'rel_score')
    # result_table.show()

    # TODO

    result_table.write.json(out_directory, mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
