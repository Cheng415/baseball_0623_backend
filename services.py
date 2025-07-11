import os
import shutil
import httpx
import asyncio
import joblib
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from config import GCS_BUCKET_NAME, POSE_API_URL, BALL_API_URL
from gcs_utils import upload_video_to_gcs
from Drawingfunction import render_video_with_pose_and_max_ball_speed, save_specific_frames
from KinematicsModule import extract_pitching_biomechanics
from PoseClassification import calculate_score_from_comparison
from BallClassification import classify_ball_quality
from typing import Dict, Optional, Tuple
import crud
logger = logging.getLogger(__name__)
API_TIMEOUT = 300

# 載入球路預測模型 用來分類好壞球
ball_prediction_model = joblib.load('random_forest_model.pkl')

# 取得比較模型 輸入資料庫 比較對象 球路 返回比較標準模型
def get_comparison_model(db: Session, benchmark_player_name: str, detected_pitch_type: str):
    profile_model = None
    if detected_pitch_type and detected_pitch_type != "Unknown":
        ideal_model_name = f"{benchmark_player_name}_{detected_pitch_type}_v1"
        logger.info(f"服務層：正在嘗試載入球種專屬模型: {ideal_model_name}")
        profile_model = crud.get_pitch_model_by_name(db, model_name=ideal_model_name)
        if profile_model:
            return profile_model

    fallback_model_name = f"{benchmark_player_name}_all_v1"
    logger.warning(f"找不到或未指定專屬模型，嘗試載入通用模型: {fallback_model_name}")
    profile_model = crud.get_pitch_model_by_name(db, model_name=fallback_model_name)
    return profile_model

# 分析生物力學特徵函數 輸入影片 返回 運動力學特徵 骨架
async def analyze_video_kinematics(video_bytes: bytes, filename: str) -> Tuple[Dict, Dict]:
    logger.info("服務層：(子任務) 正在呼叫 POSE API...")
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        files = {"file": (filename, video_bytes, "video/mp4")}
        response = await client.post(POSE_API_URL, files=files)
        response.raise_for_status()
        pose_data = response.json()
    logger.info("服務層：(子任務) 正在計算生物力學特徵...")
    biomechanics_features = extract_pitching_biomechanics(pose_data)
    return biomechanics_features, pose_data
    
# 棒球軌跡分析函數 輸入影片輸出球路軌跡
async def analyze_ball_flight(video_bytes: bytes, filename: str) -> Dict:
    """
    呼叫 Ball API 以獲取球路相關數據。
    """
    logger.info("服務層：(子任務) 正在呼叫 BALL API...")
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        files = {"file": (filename, video_bytes, "video/mp4")}
        response = await client.post(BALL_API_URL, files=files)
        response.raise_for_status()
        return response.json()

# 主要分析路由 輸入資料庫 影片 球員名稱 比較對象 返回分析結果
async def analyze_pitch_service(
        db,
        video_file, 
        player_name,
        benchmark_name,
        compare_average: bool
        ):
    
    logger.info(f"[服務層] 收到參數: player_name='{player_name}', benchmark_name='{benchmark_name}', compare_average={compare_average}") # 偵錯日誌
    
    # 步驟 1 嘗試暫存原始影片
    temp_video_path = f"temp_{video_file.filename}"
    
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
    except Exception as e:
        logger.error(f"無法儲存影片檔案: {e}", exc_info=True)
        raise e

    try:
        with open(temp_video_path, "rb") as f:
            video_bytes = f.read()
    except Exception as e:
        logger.error(f"無法讀取影片內容: {e}", exc_info=True)
        raise e

    # 步驟 2: 並行呼叫 API 分析骨架跟球路
    (kinematics_results, ball_data) = await asyncio.gather(
            analyze_video_kinematics(video_bytes, video_file.filename),
            analyze_ball_flight(video_bytes, video_file.filename)
            )
    
    # 從kinematics_results拿出骨架資料跟運動力學特徵
    biomechanics_features, pose_data = kinematics_results
    
    # 從球路資料拿到pitch_type
    detected_pitch_type = ball_data.get("predicted_pitch_type",None)
    
    # 步驟 3: 決定比較標竿並取得模型
    # 建立一個列表來存放所有要比對的模型
    benchmark_profiles_to_return = []

    # 處理菁英選手模型
    if benchmark_name:
        elite_model = crud.get_pitch_model_by_name(db, model_name=benchmark_name)
        if elite_model:
            benchmark_profiles_to_return.append(elite_model)

    # 如果勾選了，處理個人歷史平均模型
    if compare_average:
        current_time = datetime.now(timezone.utc)
        user_average_model = crud.calculate_user_average_profile(db, player_name, end_date=current_time)
        if user_average_model:
            benchmark_profiles_to_return.append(user_average_model)

    # 使用第一個模型（通常是菁英模型）來計算主要分數
    pose_score = 0
    pose_score_details = {}
    pose_score_message = "分析成功" # 預設訊息

    if benchmark_profiles_to_return:
        main_profile_data = benchmark_profiles_to_return[0].profile_data
        if main_profile_data:
            pose_score, pose_score_details = calculate_score_from_comparison(
                features=biomechanics_features,
                profile_data=main_profile_data
            )
        else:
            # 雖然有模型，但模型沒有資料的情況
            pose_score_message = "比對模型資料不完整"
            logger.warning(f"服務層：模型 {benchmark_profiles_to_return[0].model_name} 資料不完整。")
    else:
        # 【建議優化】: 在找不到模型時，更新訊息內容
        pose_score_message = "未選擇或找不到比對模型"
        logger.warning(f"服務層：找不到任何比對模型，pose_score 設為 0。")

    # 計算投球分數
    ball_score = classify_ball_quality(ball_data, ball_prediction_model)
        
    # 渲染影片
    try:
        rendered_video_local_path, max_speed_kmh = render_video_with_pose_and_max_ball_speed(
            input_video_path=temp_video_path,
            pose_json=pose_data,
            ball_json=ball_data
        )
    except Exception as e:
        logger.error(f"影片渲染失敗: {e}", exc_info=True)
        raise e

    # 上傳至 GCS
    gcs_video_url = None
    try:
        destination_blob_name = f"render_videos/rendered_{video_file.filename}"
        gcs_video_url = upload_video_to_gcs(
            bucket_name=GCS_BUCKET_NAME,
            source_file_path=rendered_video_local_path,
            destination_blob_name=destination_blob_name
        )
    except Exception as e:
        logger.error(f"GCS 上傳失敗: {e}", exc_info=True)
        raise e

    # 儲存關鍵影格圖片
    release_frame_url = None
    landing_frame_url = None
    shoulder_frame_url = None
    frame_indices = {
        "release": biomechanics_features.get("release_frame"),
        "landing": biomechanics_features.get("landing_frame"),
        "shoulder": biomechanics_features.get("shoulder_frame")
        }
        
    saved_frame_paths = save_specific_frames(temp_video_path, frame_indices)

    # 上傳關鍵影格圖片到 GCS
    try:
        if "release_frame_path" in saved_frame_paths:
            release_frame_url = upload_video_to_gcs(
                bucket_name=GCS_BUCKET_NAME,
                source_file_path=saved_frame_paths["release_frame_path"],
                destination_blob_name=f"key_frames/release_{os.path.basename(saved_frame_paths['release_frame_path'])}"
            )
        if "landing_frame_path" in saved_frame_paths:
            landing_frame_url = upload_video_to_gcs(
                bucket_name=GCS_BUCKET_NAME,
                source_file_path=saved_frame_paths["landing_frame_path"],
                destination_blob_name=f"key_frames/landing_{os.path.basename(saved_frame_paths['landing_frame_path'])}"
            )
        if "shoulder_frame_path" in saved_frame_paths:
            shoulder_frame_url = upload_video_to_gcs(
                bucket_name=GCS_BUCKET_NAME,
                source_file_path=saved_frame_paths["shoulder_frame_path"],
                destination_blob_name=f"key_frames/shoulder_{os.path.basename(saved_frame_paths['shoulder_frame_path'])}"
            )
    except Exception as e:
        logger.error(f"GCS 上傳失敗: {e}", exc_info=True)
        raise e

    # 清理本地臨時檔案
    try:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(rendered_video_local_path):
            os.remove(rendered_video_local_path)
        for key in ["release_frame_path", "landing_frame_path", "shoulder_frame_path"]:
            path = saved_frame_paths.get(key)
            if path and os.path.exists(path):
                os.remove(path)
    except Exception as e:
        logger.warning(f"刪除暫存影片失敗: {e}", exc_info=True)

    # 步驟 6: 組裝一個「扁平化」的字典，用來存入資料庫
    data_for_db = {
        "output_video_url": gcs_video_url,
        "player_name": player_name,
        "max_speed_kmh": max_speed_kmh,
        "pose_score": pose_score,
        "ball_score": ball_score,
        "biomechanics_features": biomechanics_features,
        "release_frame_url": release_frame_url,
        "landing_frame_url": landing_frame_url,
        "shoulder_frame_url": shoulder_frame_url,
        "pose_score_message": pose_score_message
    }

    # 步驟 7: 將本次分析結果存入資料庫
    try:
        created_record_from_db = crud.create_pitch_analysis(
            db=db,
            analysis_data=data_for_db
        )
        new_record_id = created_record_from_db.id
        new_record_created_at = created_record_from_db.created_at.isoformat()
        logger.info(f"成功將分析結果 (ID: {new_record_id}) 存入資料庫。")
    except Exception as e:
        logger.error(f"服務層：分析結果存入資料庫失敗: {e}", exc_info=True)
        new_record_id = None
        new_record_created_at = None

    final_response_package = {
        "new_record": {
            "id": new_record_id,
            "created_at": new_record_created_at,
            "player_name": player_name,
            "video_path": gcs_video_url,
            "keyframe_urls": {
                "release_frame_url": release_frame_url,
                "landing_frame_url": landing_frame_url,
                "shoulder_frame_url": shoulder_frame_url
            },
            "predictions": {
                "max_speed_kmh": max_speed_kmh,
                "pose_score": pose_score,
                "ball_score": ball_score,
                "pose_score_details": pose_score_details,
                "pose_score_message": pose_score_message 
            },
            "biomechanics_features": biomechanics_features
        },
        "benchmark_profiles": [
            {
                "model_name": p.model_name,
                "display_name": p.display_name if hasattr(p, 'display_name') else p.model_name,
                "profile_data": p.profile_data
            } for p in benchmark_profiles_to_return
        ]
    }

    return final_response_package