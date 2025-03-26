# Job Recommendation System

## Overview
The **Job Recommendation System** is designed to match job seekers with relevant job postings based on their resumes. The system leverages machine learning and natural language processing techniques to analyze job descriptions and resumes, generating personalized job recommendations.

## Features
- **Automated Data Retrieval**: Uses Apache Airflow DAGs to fetch job postings from APIs and process resumes.
- **Data Storage & Processing**: Stores job and resume data in MongoDB and utilizes Spark SQL for efficient preprocessing and aggregations.
- **Machine Learning Models**:
  - A neural network that encodes resumes and job descriptions into a 2D matrix representation.
  - BERT-based text embeddings for skill matching and job relevance scoring.
  - Cosine similarity for ranking job postings.
- **Google Cloud Integration**: Data and models are processed using Google Cloud services.

## Tech Stack
- **Languages**: Python, SQL
- **Data Processing**: PySpark, Spark SQL
- **Machine Learning**: PyTorch, BERT Transformers
- **Orchestration**: Apache Airflow
- **Storage**: MongoDB, Google Cloud Storage (GCS)
- **APIs Used**: Jooble API for job postings, Kaggle for resume dataset

## Architecture
1. **Data Ingestion**: Job postings retrieved via API; resume data sourced from Kaggle.
2. **Data Processing**:
   - Job postings and resume data stored in MongoDB.
   - Spark SQL used for filtering and preprocessing.
3. **Feature Engineering**:
   - Resume and job descriptions transformed into vector representations.
   - Neural network predicts match scores.
4. **Job Matching & Recommendation**:
   - Cosine similarity computed on BERT-based text embeddings.
   - Weighted scoring mechanism applied to job postings.
   - Top 10 job recommendations generated for each resume.
5. **Future Enhancements**:
   - Dynamic resume updates.
   - Reinforcement learning for improving recommendations.
   - Industry-specific weighting and hiring trends incorporation.

## Challenges & Learnings
- **Data Limitations**: Limited API access led to reliance on Jooble and Kaggle.
- **Computational Costs**: Google Cloud credits restricted full automation of ML models.
- **Collaboration**: Weekly meetings and structured task delegation helped maintain progress.

## Future Work
- Fully automating the ML pipeline within Google Cloud.
- Enhancing recommendation accuracy with real-world hiring trends.
- Improving resume parsing for better feature extraction.

---
**Contributors:** Jessica Lee, Kavin Indirajith, Maxwell Guevarra, Venkatachalam, Nathan Holmes-King
