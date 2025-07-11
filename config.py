import os

# GCS 設定
GCS_BUCKET_NAME = "baseball_cloud_storage"

# 外部 API 端點
POSE_API_URL = "https://mmpose-api-new-924124779607.europe-west1.run.app/pose_video"
BALL_API_URL = "https://base-ball-detect-api-1069614647348.us-east4.run.app/predict"

# Render PostgreSQL 資料庫 URL
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://baseball_0623_postgres_user:5EcgbNjxL90WgsAWGU5xNylqYoEvNBWx@dpg-d1d1nrmmcj7s73fai6k0-a.oregon-postgres.render.com/baseball_0623_postgres")
# Cloud SQL
# DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:baseball000@34.66.34.45:5432/postgres")


# LemonAPI
# POSE_API_URL = "http://localhost:8000/pose_video"
# BALL_API_URL = "http://localhost:8080/predict"
