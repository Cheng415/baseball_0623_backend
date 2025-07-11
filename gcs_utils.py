from google.cloud import storage

def upload_video_to_gcs(bucket_name, source_file_path, destination_blob_name):
    # 初始化 GCS 客戶端
    # 本地端使用金鑰
    #client = storage.Client.from_service_account_json("bustling-joy-463213-u1-c462f540d76c.json")
    # cloud run不需要金鑰
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # 上傳影片
    blob.upload_from_filename(source_file_path)
    print(f"影片已上傳至 gs://{bucket_name}/{destination_blob_name}")

    # 設定檔案為公開
    blob.make_public()
    print("影片已設為公開")

    # 回傳公開 URL
    public_url = blob.public_url
    print(f"公開網址：{public_url}")
    return public_url

