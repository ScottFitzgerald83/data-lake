import configparser
from datetime import datetime
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, udf, to_timestamp
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format

config = configparser.ConfigParser()
config.read_file(open("dl.cfg"))
os.environ["AWS_ACCESS_KEY_ID"] = config.get("AWS", "AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = config.get("AWS", "AWS_SECRET_ACCESS_KEY")

DESTINATION_S3_BUCKET = config.get('AWS', 'S3_BUCKET')
DATA_SOURCE_S3_BUCKET = "udacity-dend/"


def create_spark_session(debug=False):
    if debug:
        from pyspark import SparkContext
        sc = SparkContext()
        sc.setLogLevel("DEBUG")
    return SparkSession.builder \
        .appName("Sparkify data lake") \
        .getOrCreate()


def process_song_data(spark, input_data, output_data):
    """
    Uses a Spark Session to read json song data and write songs and artists tables to S3 in parquet format
    :param spark: SparkSession
    :param input_data: Directory where the data resides
    :param output_data: The directory where the parquet is being written
    :return:
    """
    s3_path = output_data
    song_files = f"{input_data}/song_data/*/*/*/*.json"
    song_data_df = spark.read.json(song_files)

    # build and write songs table to s3, partitioned by year and artist id
    song_data_df.select \
        ("song_id", "title", "artist_id", "year", "duration") \
        .dropDuplicates() \
        .write \
        .mode("overwrite") \
        .partitionBy("year", "artist_id") \
        .parquet(f"{s3_path}/songs")

    # build and write artists table to s3
    song_data_df.select \
        ("artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude") \
        .dropDuplicates() \
        .write \
        .mode("overwrite") \
        .parquet(f"{s3_path}/artists")

    return song_data_df


def process_log_data(spark, input_data, song_data_df, output_data):
    """
    Uses a Spark Session to read json log data and write users, time, and songplays tables to S3 in parquet format
    :param spark: SparkSession
    :param input_data: Directory where the data resides
    :param song_data_df: The song dataframe, used in a join on the log data
    :param output_data: The directory where the parquet is being written
    :return:
    """
    s3_path = output_data
    log_files = f"{input_data}/log_data"
    log_data_df = spark.read.json(log_files).filter("page = 'NextSong'")

    # Build and write users table to s3
    log_data_df \
        .select("userId", "firstName", "lastName", "gender", "level") \
        .dropDuplicates() \
        .write \
        .mode("overwrite") \
        .parquet(f"{s3_path}/users")

    @udf("string")
    def ts_from_epoch(epoch):
        """Converts unix epoch timestamp with millis to string"""
        return datetime.fromtimestamp(epoch / 1000).strftime("%Y-%m-%d %H:%M:%S")

    # Build and write time table to s3, partitioned by year and month
    log_data_df \
        .select(to_timestamp(ts_from_epoch("ts"), "yyyy-MM-dd HH:mm:ss").alias("ts")) \
        .select("ts", hour("ts").alias("hour"),
                dayofmonth("ts").alias("day"),
                weekofyear("ts").alias("week"),
                month("ts").alias("month"),
                year("ts").alias("year"),
                date_format("ts", "w").alias("weekday")) \
        .withColumnRenamed("ts", "start_time") \
        .write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet(f"{s3_path}/time")

    # aliases, conditions, and datetime format for songplays join
    e = log_data_df.alias("e")
    s = song_data_df.alias("s")
    join_conditions = [e.artist == s.artist_name, e.song == s.title]
    datetime_format = "yyyy-MM-dd HH:mm:ss"

    # Build and write songplays table partitioned by year and month
    # join events data on songs data to get song and artist ids
    e.join(s, join_conditions, how="left") \
        .withColumn("songplay_id", monotonically_increasing_id()).select(
        "songplay_id",
        to_timestamp(ts_from_epoch("ts"), datetime_format).alias("start_time"),
        year(ts_from_epoch("ts")).alias("year"),
        month(ts_from_epoch("ts")).alias("month"),
        "userId",
        "level",
        "song_id",
        "artist_id",
        "sessionId",
        "location",
        "userAgent") \
        .withColumnRenamed("userId", "user_id") \
        .withColumnRenamed("sessionId", "session_id") \
        .withColumnRenamed("userAgent", "user_agent") \
        .dropDuplicates() \
        .write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet(f"{s3_path}/songplays")


def main():
    spark = create_spark_session(debug=True)
    input_data = f"s3a://{DATA_SOURCE_S3_BUCKET}"
    output_data = f"s3a://{DESTINATION_S3_BUCKET}"

    song_data = process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, song_data, output_data)


if __name__ == "__main__":
    main()
