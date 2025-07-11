import os
from sqlalchemy import (create_engine, Column, Integer, String, Float, JSON,
                        DateTime, ForeignKey)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func
from sqlalchemy.exc import SQLAlchemyError

from config import DATABASE_URL

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- 最終的四表架構模型定義 ---

# 表一：儲存「訓練用」的原始投球紀錄
class PitchRecording(Base):
    __tablename__ = 'pitch_record'
    
    id = Column(Integer, primary_key=True, index=True)
    player_name = Column(String, index=True)
    pitch_type = Column(String)
    video_filename = Column(String, unique=True, index=True)
    description = Column(String)
    source_csv = Column(String)
    keypoints_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    kinematics = relationship("Kinematics", back_populates="pitch_recording", cascade="all, delete-orphan")

# 表二：儲存「訓練用」的計算後運動學特徵
class Kinematics(Base):
    __tablename__ = 'kinematics'
    
    id = Column(Integer, primary_key=True, index=True)
    pitch_record_id = Column(Integer, ForeignKey('pitch_record.id'), nullable=False)
    
    trunk_flexion_excursion = Column(Float)
    pelvis_obliquity_at_fc = Column(Float)
    trunk_rotation_at_br = Column(Float)
    shoulder_abduction_at_br = Column(Float)
    trunk_flexion_at_br = Column(Float)
    trunk_lateral_flexion_at_hs = Column(Float)
    release_frame = Column(Integer)
    landing_frame = Column(Integer)
    shoulder_frame = Column(Integer)
    total_frames = Column(Integer)

    pitch_recording = relationship("PitchRecording", back_populates="kinematics")

# 表三：儲存來自 API 的單次分析整合結果
class PitchAnalyses(Base):
    __tablename__ = "pitch_analyses"

    id = Column(Integer, primary_key=True, index=True)
    video_path = Column(String, index=True)
    player_name = Column(String, index=True)
    max_speed_kmh = Column(Float)
    pose_score = Column(Integer)
    ball_score = Column(Float)
    biomechanics_features = Column(JSON)
    release_frame_url = Column(String, index=True)
    landing_frame_url = Column(String, index=True)
    shoulder_frame_url = Column(String, index=True)
    pose_score_message = Column(String, default="分析成功")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
# 表四：儲存計算後的統計模型
class PitchModel(Base):
    __tablename__ = 'pitch_model'

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True, index=True, nullable=False)
    method = Column(String, default='percentile')
    profile_data = Column(JSON, nullable=False)
    source_feature_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# --- 3. 執行資料庫操作的函式 ---

def get_db():
    """
    FastAPI 的依賴注入函式，用來管理每個請求的資料庫連線生命週期。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def reset_database():
    """
    先刪除所有已知的資料表，然後再根據上面的模型全部重建。
    【危險操作】
    """
    print("⚠️ 警告：此操作將會刪除您 GCP 資料庫中所有已知的資料表！")
    user_confirmation = input("這將會重置資料庫為最新的四表架構，請輸入 'yes' 以確認執行: ")

    if user_confirmation.lower() != 'yes':
        print("操作已取消。")
        return

    try:
        print("正在刪除舊的資料表...")
        Base.metadata.drop_all(bind=engine)
        
        print("正在根據新的藍圖建立資料表...")
        Base.metadata.create_all(bind=engine)
        
        print("✅ 資料庫已成功重置為最新的四表架構！")
    except Exception as e:
        print(f"❌ 重置資料庫時發生錯誤: {e}")


def reset_single_table(model_class):
    """
    只重置 (刪除並重建) 指定的單一資料表。
    
    :param model_class: 要重置的 SQLAlchemy 模型類別 (例如: PitchModel)
    """
    table_name = model_class.__tablename__
    print(f"⚠️ 警告：此操作將會刪除並重建 '{table_name}' 資料表！")
    user_confirmation = input(f"請輸入 '{table_name}' 以確認執行: ")

    if user_confirmation != table_name:
        print("操作已取消。")
        return

    try:
        target_table = model_class.__table__
        print(f"正在刪除 '{table_name}' 資料表...")
        # checkfirst=True 可避免在資料表不存在時拋出錯誤
        target_table.drop(bind=engine, checkfirst=True)
        
        print(f"正在重建 '{table_name}' 資料表...")
        target_table.create(bind=engine)
        
        print(f"✅ '{table_name}' 資料表已成功重置！")
    except Exception as e:
        print(f"❌ 操作 '{table_name}' 資料表時發生錯誤: {e}")


if __name__ == "__main__":
    print("您想要執行哪個操作？")
    print("1. 重置 'pitch_profiles' 資料表 (推薦)")
    print("2. 重置整個資料庫 (危險操作！)")
    choice = input("請輸入選項 (1/2): ")

    if choice == '1':
        reset_single_table(PitchModel)
    elif choice == '2':
        reset_database()
    else:
        print("無效的選項，操作已取消。")
