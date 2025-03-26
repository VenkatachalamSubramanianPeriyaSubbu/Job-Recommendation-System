from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import http.client
import json
import time
import os
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


# Google Cloud Storage & MongoDB Atlas Config
GCP_BUCKET_NAME = "us-central1-dds-project-com-c71679f7-bucket"  # Replace with your GCS bucket
GCP_JSON_FILENAME = "jooble_jobs_all_states.json"
MONGO_URI = "mongodb+srv://ddsproject:ddsproject@cluster1.5x6jl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1" # MongoDB Atlas connection
DB_NAME = "job_database"
COLLECTION_NAME = "jooble_jobs_new_collection"

with DAG(dag_id="jooble_dag",
         start_date=datetime(2025, 2, 21),
         end_date=datetime(2025, 12, 31),
         schedule="@once") as dag:

    def py_function():
        api_key = "a90cdd1e-9bd7-4e4c-a159-916cf175b97a"
        host = "jooble.org"

        connection = http.client.HTTPSConnection(host)

        headers = {"Content-type": "application/json"}

        # List of all U.S. states + Washington D.C.
        locations = [
            "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
            "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
            "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
            "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
            "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
            "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
            "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming", "Washington D.C."
        ]

        all_jobs = []

        for location in locations:
            job_filters = {
                "location": location,
                "companysearch": "false",
                "page": 1,
                "resultOnPage": 100  # Fetch up to 100 results per request
            }

            print(f"\nFetching jobs for: {location}")

            while True:
                # Convert request body to JSON
                body = json.dumps(job_filters)

                # Make API request
                connection.request('POST', f'/api/{api_key}', body, headers)
                response = connection.getresponse()

                if response.status == 200:
                    data = json.loads(response.read())

                    # If no jobs are returned, stop for this location
                    if "jobs" in data and data["jobs"]:
                        for job in data["jobs"]:
                            transformed_job = {
                                "id": job.get("id", ""),
                                "title": job.get("title", ""),
                                "company": job.get("company", ""),
                                "location": location,  # Ensure location is stored
                                "date": job.get("updated", ""),
                                "url": job.get("link", ""),
                                "description": job.get("snippet", ""),
                                "salary": job.get("salary", ""),
                                "source": job.get("source", ""),
                                "type": job.get("type", ""),
                            }
                            all_jobs.append(transformed_job)

                        print(
                            f"Collected {len(data['jobs'])} jobs from page {job_filters['page']} (Total so far: {len(all_jobs)})")
                    else:
                        print(
                            f"No more jobs found for {location}. Moving to next state.")
                        break  # Stop pagination for this location

                    # Move to the next page
                    job_filters["page"] += 1

                    # Add a small delay to avoid hitting API rate limits
                    time.sleep(1)
                else:
                    print(
                        f"API Error: {response.status} {response.reason} for {location}")
                    break  # Stop on API failure

        # Save all jobs into a single JSON file
        output_file = "jooble_jobs_all_states.json"
        full_path = os.path.abspath(output_file)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_jobs, f, indent=4)

        print(f"\nFinished fetching jobs from all states.")
        print(f"Total jobs saved: {len(all_jobs)}")
        print(f"All jobs saved in: {full_path}")

    def mongo_function():
        """Load JSON and insert unique jobs into MongoDB."""
        
        MONGO_URI = "mongodb+srv://ddsproject:ddsproject@cluster1.5x6jl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
        DB_NAME = "job_database"
        COLLECTION_NAME = "jooble_jobs_new_collection"

        # Use correct JSON path
        JSON_FILE_PATH = os.path.abspath("jooble_jobs_all_states.json")

        # Connect to MongoDB
        # Create a new client and connect to the server
        # client = MongoClient(uri, server_api=ServerApi('1'))

        # # Send a ping to confirm a successful connection
        # try:
        #     client.admin.command('ping')
        #     print("Pinged your deployment. You successfully connected to MongoDB!")
        # except Exception as e:
        #     print(e)
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Check if JSON file exists
        if not os.path.exists(JSON_FILE_PATH):
            print(f"JSON file not found at {JSON_FILE_PATH}")
            return

        # Load JSON data
        try:
            with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                jobs_data = json.load(f)
        except json.JSONDecodeError:
            print("Error reading JSON file. Ensure it's valid JSON.")
            return  # Prevents further execution if JSON is invalid

        # Insert only unique jobs (Avoid duplicates based on 'id')
        inserted_count = 0
        for job in jobs_data:
            job_id = job.get("id")
            if job_id and not collection.find_one({"id": job_id}):  # Insert only if ID doesn't exist
                collection.insert_one(job)
                inserted_count += 1

        print(f"Inserted {inserted_count} new jobs into MongoDB (Skipping duplicates).")


    task1 = PythonOperator(task_id="task1",
                           python_callable=py_function,)
    task2 = PythonOperator(task_id="task2",
                           python_callable=mongo_function,)
    task1 >> task2
