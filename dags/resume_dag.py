from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import matplotlib.pyplot as plt
from pymongo import MongoClient
import pandas as pd
import numpy as np
import csv
import os
from google.cloud import storage
import io
import logging

# Connection
MONGO_URI = "mongodb+srv://ddsproject:ddsproject@cluster1.5x6jl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
DB = "job_database"
COLLECTION = "resume_data"
# Fix the bucket name format - remove the path part from the bucket name
CLOUD_BUCKET_NAME = "us-central1-dds-project-com-655f3d23-bucket"
CLOUD_PATH = "data"  # Separate the path

def perform_aggregations_get_charts_and_store_in_gsbucket():
    """
    Performs MongoDB aggregations based on predefined queries,
    creates bar charts from the results, and uploads them to Google Cloud Storage.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting aggregation and chart generation process")
    
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB]
        logger.info(f"Connected to MongoDB database: {DB}")
        
        # Connect to Google Cloud Storage
        gs_client = storage.Client()
        try:
            gs_bucket = gs_client.get_bucket(CLOUD_BUCKET_NAME)
            logger.info(f"Connected to GCS bucket: {CLOUD_BUCKET_NAME}")
        except Exception as e:
            logger.error(f"Error accessing bucket {CLOUD_BUCKET_NAME}: {str(e)}")
            # Try listing available buckets to help debugging
            buckets = list(gs_client.list_buckets())
            logger.info(f"Available buckets: {[b.name for b in buckets]}")
            raise
        
        # Query Pipelines
        AGG_QUERIES = {
            "Top Skills": [
                {"$project": {"skills": "$skills"}},
                {"$unwind": "$skills"},
                {"$group": {"_id": "$skills", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5}
            ],
            "Top Language Known": [
                {"$project": {"languages": "$languages"}},
                {"$unwind": "$languages"},
                {"$group": {"_id": "$languages.languages", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5}
            ],
            "English Proficiency Level": [
                {"$project": {"languages": "$languages"}},
                {"$unwind": "$languages"},
                {"$match": {"languages.languages": "English"}},
                {"$group": {"_id": "$languages.proficiency_levels", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ],
            "Major Field of Study": [
                {"$project": {"education": "$education"}},
                {"$unwind": "$education"},
                {"$group": {"_id": "$education.major_field_of_studies", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5},
                {"$match": {"_id": {"$ne": "N/A", "$ne": None}}}
            ],
            "Popular Education-Degree": [
                {"$unwind": "$education"},
                {"$group": {"_id": "$education.degree_names", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5}
            ]
        }

        for collection, agg_pipeline in AGG_QUERIES.items():
            try:
                logger.info(f"Running aggregation for: {collection}")
                result = list(db[COLLECTION].aggregate(agg_pipeline))
                logger.info(f"Got {len(result)} results for {collection}")
                
                df = pd.DataFrame(result)
                if not df.empty: 
                    df = df.dropna(subset=['_id'])
                    df['_id'] = df['_id'].astype(str)
                    df = df.sort_values('count', ascending=False)
                    
                    # Create chart
                    fig, ax = plt.subplots(figsize=(20, 10))
                    ax.bar(df['_id'], df['count'], color='skyblue')
                    ax.set_xlabel("Category", fontsize=12)
                    ax.set_ylabel("Count", fontsize=12)
                    ax.set_title(f"{collection}", fontsize=16)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
            

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    
                    blob_path = f"{CLOUD_PATH}/{collection.replace(' ', '_')}_chart.png"
                    blob = gs_bucket.blob(blob_path)
                    blob.upload_from_file(buf, content_type='image/png')
                    logger.info(f"Successfully uploaded chart to gs://{CLOUD_BUCKET_NAME}/{blob_path}")
                    
                    # Clean up
                    buf.close()
                    plt.close(fig)
                else:
                    logger.warning(f"No data found for {collection}")
                    
            except Exception as e:
                logger.error(f"Error processing {collection}: {str(e)}")
                
        client.close()
        logger.info("All charts generated and uploaded successfully")
        
    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
        raise
            
# DAG
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 3, 2),
    "retries": 1,
    "email_on_failure": True,
}

dag = DAG(
    "mongo_to_cloud_img_dag",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    description="Generate charts from MongoDB resume data and upload to GCS",
    tags=["mongodb", "charts", "resume"]
)

task_export = PythonOperator(
    task_id="export_img_to_cloud",
    python_callable=perform_aggregations_get_charts_and_store_in_gsbucket,
    dag=dag
)

task_export