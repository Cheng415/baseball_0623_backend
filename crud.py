# 檔案: crud.py
# 職責: 作為資料庫的唯一接口 (數據庫管家)，提供所有資料的增刪改查功能。

import numpy as np
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from collections import defaultdict
from types import SimpleNamespace
from datetime import datetime

from database import PitchAnalyses, PitchModel
from models import PitchAnalysisUpdate


# --- 針對 PitchAnalyses (單次測試結果) 的操作 ---

def get_pitch_analysis(db: Session, analysis_id: int) -> Optional[PitchAnalyses]:
    """根據 ID 獲取單筆分析紀錄。"""
    return db.query(PitchAnalyses).filter(PitchAnalyses.id == analysis_id).first()

def get_pitch_analyses(db: Session, player_name: Optional[str] = None, end_date: Optional[datetime] = None, skip: int = 0, limit: int = 100) -> List[PitchAnalyses]:
    """
    獲取分析紀錄列表，可選擇性地根據投手名稱和結束時間篩選。
    """
    
    query = db.query(PitchAnalyses).order_by(PitchAnalyses.id.desc())
    if player_name:

        query = query.filter(PitchAnalyses.player_name == player_name)
    
    if end_date:
        query = query.filter(PitchAnalyses.created_at < end_date)
        
    return query.offset(skip).limit(limit).all()

def create_pitch_analysis(db: Session, analysis_data: Dict[str, Any]) -> PitchAnalyses:
    """
    根據傳入的字典，建立一筆新的分析紀錄。
    """
    db_analysis = PitchAnalyses(
        video_path=analysis_data.get("output_video_url"),
        player_name=analysis_data.get("player_name"), # 【修改點】
        max_speed_kmh=analysis_data.get("max_speed_kmh"),
        pose_score=analysis_data.get('pose_score'), # 【修改點】
        ball_score=analysis_data.get('ball_score'),
        biomechanics_features=analysis_data.get("biomechanics_features"),
        release_frame_url=analysis_data.get("release_frame_url"),
        landing_frame_url=analysis_data.get("landing_frame_url"),
        shoulder_frame_url=analysis_data.get("shoulder_frame_url"),
        pose_score_message=analysis_data.get("pose_score_message", "分析成功")
    )
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis

def update_pitch_analysis(db: Session, analysis_id: int, updated_data: PitchAnalysisUpdate) -> Optional[PitchAnalyses]:
    """更新指定的分析紀錄。"""
    db_analysis = get_pitch_analysis(db, analysis_id)
    if db_analysis:
        # exclude_unset=True 表示只更新前端有提供的欄位
        for key, value in updated_data.dict(exclude_unset=True).items():
            setattr(db_analysis, key, value)
        db.commit()
        db.refresh(db_analysis)
    return db_analysis


# --- 針對 PitchModel (統計模型) 的操作 ---

def get_all_pitch_models(db: Session) -> List[PitchModel]:
    """
    從資料庫中獲取所有 PitchModel 紀錄。
    """
    return db.query(PitchModel).order_by(PitchModel.model_name).all()

def get_pitch_model_by_name(db: Session, model_name: str) -> Optional[PitchModel]:
    """
    根據模型名稱，從 pitch_model 資料表中查詢一個統計模型。
    """
    return db.query(PitchModel).filter(PitchModel.model_name == model_name).first()

def calculate_user_average_profile(db: Session, player_name: str, end_date: Optional[datetime] = None) -> Optional[SimpleNamespace]:
    """
    即時計算指定投手在某個時間點之前的歷史平均數據。
    """
    # 1. 獲取該選手在指定時間點之前的歷史分析紀錄
    history_records = get_pitch_analyses(db, player_name=player_name, end_date=end_date)

    if not history_records or len(history_records) < 1:
        return None

    # 2. 收集每個生物力學特徵的所有歷史數值
    feature_values: Dict[str, List[float]] = defaultdict(list)
    
    for record in history_records:
        if record.biomechanics_features:
            for key, value in record.biomechanics_features.items():
                if isinstance(value, (int, float)):
                    feature_values[key].append(value)
    
    if not feature_values:
        return None

    # 3. 計算統計數據
    profile_data = {}
    for feature, values in feature_values.items():
        np_values = np.array(values)
        profile_data[feature.lower()] = {
            "mean": np.mean(np_values),
            "std": np.std(np_values),
            "min": float(np.min(np_values)),
            "max": float(np.max(np_values)),
            "p10": float(np.percentile(np_values, 10)),
            "p50_median": float(np.median(np_values)),
            "p90": float(np.percentile(np_values, 90)),
        }

    # 4. 打包成臨時模型物件
    user_average_model = SimpleNamespace(
        model_name=f"{player_name} 個人歷史平均",
        display_name=f"{player_name} 個人歷史平均",
        profile_data=profile_data
    )

    return user_average_model