# 檔案: mainV2.py
# 職責: 作為 API 的入口點，接收請求並完全轉交給服務層處理。

import logging
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Query, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
import uvicorn

# --- 從我們的「資料庫中心」和「服務中心」匯入 ---
# 已更新為您最新的駝峰式檔名
from crud import create_pitch_analysis
import crud
import services
from database import get_db, PitchAnalyses
from models import PitchAnalysisUpdate

# --- 全域設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API 路由 ---

@app.post("/analyze-pitch/")
async def analyze_pitch(
    db: Session = Depends(get_db),
    video_file: UploadFile = File(...), 
    player_name: str = Form(...),
    benchmark_name: str = Form(...),
    compare_average: bool = Form(False)
):
    """
    接收前端請求，將所有工作轉交給服務層，並直接回傳服務層的結果。
    """
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="未上傳影片檔案")

    try:
        final_response_package = await services.analyze_pitch_service(
            db=db,
            video_file=video_file,
            player_name=player_name,
            benchmark_name=benchmark_name,
            compare_average=compare_average 
        )
        
        return final_response_package

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"影片分析處理失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"影片分析處理失敗: {str(e)}")


@app.get("/history/")
async def get_history_analyses(player_name: str = None, db: Session = Depends(get_db)):
    try:
        history_records = crud.get_pitch_analyses(db, player_name)
        return [
            {
                "id": record.id,
                "video_path": record.video_path,
                "max_speed_kmh": record.max_speed_kmh,
                "pose_score": record.pose_score,
                "ball_score": record.ball_score,
                "biomechanics_features": record.biomechanics_features,
                "player_name": record.player_name,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "pose_score_message": getattr(record, 'pose_score_message', '分析成功'),
                "keyframe_urls": {
                    "release_frame_url": record.release_frame_url or "",
                    "landing_frame_url": record.landing_frame_url or "",
                    "shoulder_frame_url": record.shoulder_frame_url or ""
                }
            }
            for record in history_records
        ]
    except SQLAlchemyError as e:
        logger.error(f"無法獲取歷史紀錄: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"無法獲取歷史紀錄: {str(e)}")

@app.get("/models/")
async def get_available_models(db: Session = Depends(get_db)):
    """
    提供前端動態建立「比對標竿」下拉選單所需的所有菁英選手模型，
    並包含完整的 profile_data 供前端快取使用。
    """
    try:
        models_from_db = crud.get_all_pitch_models(db)
        
        # 建立一個字典來翻譯球種縮寫
        pitch_type_translator = {
            "FS": "分指快速球 / 指叉球 (Splitter / Split-Finger Fastball)",
            "FF": "四縫線快速球 (Four-Seam Fastball)",
            "SL": "滑球 (Slider)",
            "CU": "曲球 (Curveball)",
            "CH": "變速球 (Changeup)",
            "FO": "指叉球 (Forkball)",
            "all": "通用"
        }
        
        formatted_models = []
        for model in models_from_db:
            parts = model.model_name.split('_')
            # 預期格式為 "姓, 名_球種縮寫_v1" 或 "名字_姓氏_球種縮寫_v1"
            if len(parts) >= 2:
                player_name = parts[0].replace(",", ", ")
                pitch_type_abbr = parts[1] if len(parts) > 1 else "all"
                
                # 產生顯示用的名稱
                pitch_type_display = pitch_type_translator.get(pitch_type_abbr, pitch_type_abbr)
                display_name = f"{player_name} - {pitch_type_display}"
            else:
                display_name = model.model_name # 如果格式不符，使用原名
            
            formatted_models.append({
                "model_name": model.model_name,      # 後端比對時需要的原始名稱
                "display_name": display_name,        # 前端顯示用的乾淨名稱
                "profile_data": model.profile_data   # 【重點】將完整的 profile_data 一併回傳
            })
            
        return formatted_models
        
    except SQLAlchemyError as e:
        logger.error(f"無法獲取模型列表: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"無法獲取模型列表: {str(e)}")

@app.get("/user-average-profile/{player_name}")
async def get_user_average_profile_endpoint(player_name: str, db: Session = Depends(get_db)):
    """
    即時計算並回傳指定投手的歷史平均模型。
    """
    try:
        # 在這裡，我們不傳入 end_date，代表計算該投手的所有歷史資料平均
        profile = crud.calculate_user_average_profile(db, player_name=player_name)
        if not profile:
            raise HTTPException(status_code=404, detail="該投手歷史紀錄不足，無法產生平均模型")
        
        # 將 SimpleNamespace 物件轉換為字典以便序列化
        return {
            "model_name": profile.model_name,
            "display_name": profile.display_name,
            "profile_data": profile.profile_data
        }
    except SQLAlchemyError as e:
        logger.error(f"計算個人平均失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"計算個人平均失敗: {str(e)}")

@app.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
     try:
         if not crud.delete_pitch_analysis(db, analysis_id):
             raise HTTPException(status_code=404, detail="分析紀錄未找到")
         logger.info(f"分析紀錄 ID: {analysis_id} 已成功刪除")
         return {"message": "分析紀錄已成功刪除"}
     except SQLAlchemyError as e:
         logger.error(f"刪除分析紀錄失敗: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"刪除分析紀錄失敗: {e}")

@app.put("/analyses/{analysis_id}")
async def update_analysis(analysis_id: int, updated_data: PitchAnalysisUpdate, db: Session = Depends(get_db)):
     try:
         analysis = crud.update_pitch_analysis(db, analysis_id, updated_data)
         if not analysis:
             raise HTTPException(status_code=404, detail="分析紀錄未找到")
         logger.info(f"分析紀錄 ID: {analysis_id} 已成功更新")
         return analysis
     except SQLAlchemyError as e:
         logger.error(f"更新分析紀錄失敗: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"更新分析紀錄失敗: {e}")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080)) # 建議使用一個新的埠號
    uvicorn.run("main:app", host="0.0.0.0", port=port)
