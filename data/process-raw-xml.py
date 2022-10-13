#%%
import os

if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]
# os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 8G"

# %%

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = (
    SparkSession.builder.master("local[*]")
    .appName("data-prep")
    .config("spark.jars.packages", "com.databricks:spark-xml_2.12:0.15.0")
    .getOrCreate()
)

print(spark.sparkContext.uiWebUrl)

#%%
schema = (
    spark.read.format("xml")
    .option("rootTag", "posts")
    .option("rowTag", "row")
    .option("samplingRatio", "1")
    .option("inferSchema", True)
    .option("attributePrefix", "")
    .load("data/1000Posts.xml")
).schema

#%%
df = (
    spark.read.format("xml")
    .option("rootTag", "posts")
    .option("rowTag", "row")
    .option("attributePrefix", "")
    .schema(schema)
    .load("data/Posts.xml")
)
# %%
python_questions = df.filter("PostTypeId = 1 AND tags LIKE '%<python>%'")
python_questions.write.mode("overwrite").parquet("data/questions-python-only.parquet")

# %%
answers = df.filter("PostTypeId = 2")
answers.write.mode("overwrite").parquet("data/answers-all.parquet")

# %%
questions = spark.read.parquet("data/questions-python-only.parquet")
answers = spark.read.parquet("data/answers-all.parquet")

answers.createOrReplaceTempView("answers")
python_questions.createOrReplaceTempView("questions")

qna = spark.sql(
    """
    SELECT
        q.Id AS question_id,
        q.AcceptedAnswerId AS accepted_answer_id,
        q.CreationDate AS question_date,
        q.Score as question_score,
        q.ViewCount as view_count,
        q.Body as question_body,
        q.Title as title,
        q.Tags as tags,
        q.AnswerCount as answer_count,
        q.FavoriteCount as favorite_count,
        a.CreationDate as answer_date,
        a.id as answer_id,
        a.score as answer_score,
        a.body as answer_body
    FROM answers a
    INNER JOIN questions q
    ON (q.id = a.parentid)
"""
)

qna.write.mode("overwrite").parquet("data/qna.parquet")
# %%
qna = spark.read.parquet("data/qna.parquet")
print(f"Working with {qna.count()} RAW answers.")

qna.createOrReplaceTempView("qna")

# Selecting the accepted or the highest rated answer for each question
out_df = spark.sql(
    """
    WITH ranked_qna AS (
        SELECT *,
               RANK() OVER (
                PARTITION BY question_id ORDER BY
                   CASE WHEN answer_id == accepted_answer_id THEN 999999999
                   ELSE answer_score END 
                   DESC
               ) as answer_rank
        FROM qna
    )
    SELECT *,
       CASE WHEN answer_id == accepted_answer_id THEN TRUE
            ELSE FALSE END as answer_accepted
       FROM ranked_qna WHERE answer_rank = 1 
"""
)
out_df = out_df.withColumn(
    "tags",
    F.split(
        F.regexp_replace(
            F.regexp_replace(F.regexp_replace(F.col("tags"), ",", " "), "><", ","),
            "[<>]",
            "",
        ),
        ",",
    ),
)
out_df = out_df.drop("accepted_answer_id", "answer_rank")
out_df.write.mode("overwrite").parquet("data/so-answers-python-multipart.parquet")
out_df.repartition(1).write.mode("overwrite").parquet("data/so-answers-python.parquet")
# %%
