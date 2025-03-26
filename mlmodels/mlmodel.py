from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lit, udf, struct, row_number, concat_ws, when, monotonically_increasing_id, lower, pandas_udf, array_intersect, split, size, col
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Summarizer
from pyspark.sql.types import *
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# Initialize Spark Session
spark = SparkSession.builder.appName("JobMatching").getOrCreate()

# Define schema for jobs
jobs_schema = StructType([
    StructField("id", LongType(), True),
    StructField("title", StringType(), True),
    StructField("company", StringType(), True),
    StructField("location", StringType(), True),
    StructField("date", StringType(), True),
    StructField("url", StringType(), True),
    StructField("description", StringType(), True),
    StructField("salary", StringType(), True),
    StructField("source", StringType(), True),
    StructField("type", StringType(), True)
])

# Define schema for resumes, THIS DID NOT WORK
resumes_schema = StructType([
    StructField("address", StringType(), True),
    StructField("career_objective", StringType(), True),
    StructField("skills", ArrayType(StringType()), True),
    
    # Education: Nested Struct
    StructField("education", ArrayType(StructType([
        StructField("educational_institution_name", StringType(), True),
        StructField("degree_names", StringType(), True),
        StructField("passing_years", StringType(), True),
        StructField("educational_results", StringType(), True),
        StructField("result_types", StringType(), True),
        StructField("major_field_of_studies", StringType(), True)
    ])), True),

    # Experience: Nested Struct
    StructField("experience", ArrayType(StructType([
        StructField("professional_company_names", StringType(), True),
        StructField("company_urls", StringType(), True),
        StructField("start_dates", StringType(), True),
        StructField("end_dates", StringType(), True),
        StructField("related_skils_in_job", ArrayType(StringType()), True),
        StructField("positions", StringType(), True),
        StructField("locations", StringType(), True)
    ])), True),

    StructField("extracurriculars", ArrayType(StringType()), True),
    StructField("languages", ArrayType(StringType()), True),
    StructField("certifications", ArrayType(StringType()), True),
    
    StructField("job_position_name", StringType(), True),
    StructField("educationaL_requirements", StringType(), True),
    StructField("experiencere_requirement", StringType(), True),
    StructField("age_requirement", StringType(), True),
    StructField("responsibilities.1", StringType(), True),
    StructField("skills_required", StringType(), True),
    StructField("matched_score", FloatType(), True)
])


# Read JSON with predefined schema
spark.conf.set("google.cloud.auth.service.account.json.keyfile", "dds-project-451604-d46ec6320c25.json")
jobs_df = spark.read.schema(jobs_schema).option("multiline", "true").json("gs://us-central1-dds-project-com-655f3d23-bucket/data/jooble_jobs_all_states.json")
resumes_df = spark.read.option("multiline", "true").json("gs://us-central1-dds-project-com-655f3d23-bucket/data/jooble_jobs_all_states.json")

jobs_df.show(5, truncate=False)

resumes_df.show(5, truncate=False)

# Rename the incorrect column
resumes_df = resumes_df.withColumnRenamed("﻿job_position_name", "job_position_name")

# Now check if it works
resumes_df.select("job_position_name").show(5, truncate=False)

filtered_resumes_df = resumes_df.filter(col("matched_score") > 0.85)

filtered_resumes_df.count()

filtered_resumes_df.printSchema()

# Ensure job_position_name is correctly named
filtered_resumes_df = filtered_resumes_df.withColumnRenamed("﻿job_position_name", "job_position_name")

# Split job titles and positions into word arrays
jobs_df = jobs_df.withColumn("title_words", split(col("title"), " "))
filtered_resumes_df = filtered_resumes_df.withColumn("position_words", split(col("job_position_name"), " "))

# Perform cross join between jobs and resumes
jobs_resumes_df = jobs_df.crossJoin(filtered_resumes_df)

# Compute word overlap between job titles and resume positions
jobs_resumes_df = jobs_resumes_df.withColumn("word_overlap", array_intersect(col("title_words"), col("position_words")))

# Filter jobs where at least one word overlaps
similar_jobs_df = jobs_resumes_df.filter(size(col("word_overlap")) > 0)

# Count the number of similar jobs
similar_jobs_count = similar_jobs_df.count()
print(f"Number of similar jobs found: {similar_jobs_count}")

# Create a filtered jobs DataFrame containing only the relevant columns
filtered_jobs_df = similar_jobs_df.select(
    "id", "title", "company", "location", "date", "url", "description", "salary", "source", "type"
).distinct()

# Show the filtered jobs
filtered_jobs_df.count()

# Assign unique ID to each candidate
resumes_df = filtered_resumes_df.withColumn("candidate_id", monotonically_increasing_id())
jobs_df = filtered_jobs_df.fillna("")
resumes_df = resumes_df.fillna("Unknown Position", subset=["job_position_name"])
jobs_df = jobs_df.withColumn("title", lower(col("title")))
resumes_df = resumes_df.withColumn("job_position_name", lower(col("job_position_name")))
resumes_df = resumes_df.withColumnRenamed("responsibilities.1", "responsibilities_1")
resumes_df = resumes_df.withColumn("skills_text", concat_ws(" ", col("skills")))
resumes_df.show()

# Load BERT tokenizer & model once (avoiding reloading in each row)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Define Pandas UDF for efficient BERT embedding computation
@pandas_udf(ArrayType(FloatType()))
def bert_pandas_udf(text_series: pd.Series) -> pd.Series:
    """Computes BERT embeddings in parallel using Pandas UDF."""
    embeddings = []
    for text in text_series:
        if text is None or text.strip() == "":
            embeddings.append([0.0] * 768)  # Return zero vector for missing text
        else:
            tokens = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                output = model(**tokens)
            embeddings.append(output.last_hidden_state.mean(dim=1).squeeze().tolist())
    return pd.Series(embeddings)

# Apply BERT embeddings efficiently
jobs_df = jobs_df.withColumn("bert_description", bert_pandas_udf(col("description")))
resumes_df = resumes_df.withColumn("bert_career_objective", bert_pandas_udf(col("career_objective")))
resumes_df = resumes_df.withColumn("bert_skills", bert_pandas_udf(col("skills_text")))
resumes_df = resumes_df.withColumn("bert_responsibilities", bert_pandas_udf(col("responsibilities_1")))

# Check results
jobs_df.select("description", "bert_description").show(5, truncate=False)
resumes_df.select("career_objective", "bert_career_objective").show(5, truncate=False)

# Define cosine similarity function as UDF
@pandas_udf(FloatType())
def cosine_similarity(vec1_series: pd.Series, vec2_series: pd.Series) -> pd.Series:
    """Computes cosine similarity between two BERT vectors."""
    similarities = []
    for vec1, vec2 in zip(vec1_series, vec2_series):
        if vec1 is None or vec2 is None:
            similarities.append(0.0)
        else:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            similarity = float(dot_product / (norm_a * norm_b)) if norm_a and norm_b else 0.0
            similarities.append(similarity)
    return pd.Series(similarities)

# Cross join resumes with jobs (matching all candidates with all jobs)
cross_df = resumes_df.crossJoin(jobs_df)

# Compute similarity scores using different resume components
cross_df = cross_df.withColumn("description_match", cosine_similarity(col("bert_description"), col("bert_career_objective")))
cross_df = cross_df.withColumn("skills_match", cosine_similarity(col("bert_description"), col("bert_skills")))
cross_df = cross_df.withColumn("responsibilities_match", cosine_similarity(col("bert_responsibilities"), col("bert_career_objective")))

# Compute final match score using weighted average
cross_df = cross_df.withColumn("match_score", 
    (col("description_match") * 0.4 + 
     col("skills_match") * 0.4 + 
     col("responsibilities_match") * 0.2)
)

# Rank jobs for each candidate
window_spec = Window.partitionBy("candidate_id").orderBy(col("match_score").desc())
cross_df = cross_df.withColumn("rank", row_number().over(window_spec))

# Select top 10 matches per candidate
top_matches_df = cross_df.filter(col("rank") <= 10)

# Select relevant columns and display results
final_results = top_matches_df.select("candidate_id", "id", "title", "company", "match_score")
final_results.show() 

# spark.stop()



