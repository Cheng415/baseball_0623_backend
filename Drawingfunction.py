import cv2
import os
import math
import numpy as np
from typing import Tuple # 導入 Tuple 以正確標註回傳型別

"""
畫投手骨架的函數
"""
# COCO 17 個關節點的骨架連接規則
SKELETON_CONNECTIONS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4]
]

# 為不同骨架部分定義顏色 (BGR 格式)
LIMB_COLORS = [
    (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255), (255, 128, 0),
    (255, 128, 0), (255, 128, 0), (255, 128, 0), (0, 255, 0), (0, 255, 0),
    (0, 255, 0), (0, 255, 0), (255, 0, 255), (255, 0, 255), (255, 0, 255),
    (255, 0, 255), (255, 0, 255)
]

KEYPOINT_COLOR = (0, 0, 255) # 關節點顏色
BBOX_COLOR = (0, 255, 0) # Bounding Box 顏色

def draw_pitcher_on_frame(image, pitcher_data, kpt_thr=0.3, line_thickness=1, point_radius=3):
    """
    在一幀影像上繪製一個投手的骨架和邊界框。
    """
    if not pitcher_data:
        return image

    bbox_data = pitcher_data.get('bbox')
    keypoints_data = pitcher_data.get('keypoints')
    keypoint_scores_data = pitcher_data.get('keypoint_scores')


    # API 回傳的 bbox 數據可能有多層列表包裝，例如 [[x1, y1, x2, y2]]。
    # 下面的 while 迴圈會解開這層包裝，確保我們拿到的是 [x1, y1, x2, y2] 這種可以直接使用的格式。
    # 如果這個解包邏輯不夠完善，或 bbox 數據本身有問題，繪製就會被跳過。
    while isinstance(bbox_data, list) and len(bbox_data) > 0 and isinstance(bbox_data[0], list):
        bbox_data = bbox_data[0]
    
    bbox = bbox_data
    
    # 繪製邊界框 (Bounding Box)
    if bbox and len(bbox) == 4 and all(isinstance(c, (int, float)) for c in bbox):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), BBOX_COLOR, line_thickness)

    # 檢查數據是否存在且非空
    if not keypoints_data or not keypoint_scores_data:
        return image

    keypoints = np.array(keypoints_data)
    keypoint_scores = np.array(keypoint_scores_data)
    
    # 檢查數據維度是否正確
    if keypoints.ndim != 2 or keypoints.shape[1] != 2 or keypoint_scores.ndim != 1:
        print(f"⚠️ 數據格式不正確，無法繪製骨架。Keypoints shape: {keypoints.shape}, Scores shape: {keypoint_scores.shape}")
        return image

    # 繪製骨架連接線
    for i, (p1_idx, p2_idx) in enumerate(SKELETON_CONNECTIONS):
        if p1_idx < len(keypoint_scores) and p2_idx < len(keypoint_scores):
            if keypoint_scores[p1_idx] > kpt_thr and keypoint_scores[p2_idx] > kpt_thr:
                pt1 = (int(keypoints[p1_idx][0]), int(keypoints[p1_idx][1]))
                pt2 = (int(keypoints[p2_idx][0]), int(keypoints[p2_idx][1]))
                # 線條的粗細由 line_thickness 參數控制
                cv2.line(image, pt1, pt2, LIMB_COLORS[i], line_thickness)

    # 繪製關節點
    for i, kpt in enumerate(keypoints):
        if i < len(keypoint_scores) and keypoint_scores[i] > kpt_thr:
            x, y = int(kpt[0]), int(kpt[1])
            # 點的大小 (半徑) 由 point_radius 參數控制
            cv2.circle(image, (x, y), point_radius, KEYPOINT_COLOR, -1)

    return image

def render_video_with_pose_and_max_ball_speed(input_video_path: str,
                                              pose_json: dict,
                                              ball_json: dict,
                                              pixel_to_meter: float = 0.04,
                                              min_valid_speed_kmh: float = 30,
                                              max_valid_speed_kmh: float = 200) -> Tuple[str, float]:
    # Define a temporary directory for rendered videos
    output_dir = "temp_rendered_videos"
    os.makedirs(output_dir, exist_ok=True) # Ensure this directory exists

    # Construct the output video path within the temporary directory
    output_video_path = os.path.join(output_dir, f"temp_rendered_{os.path.basename(input_video_path)}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #fourcc = cv2.VideoWriter_fourcc(*'X264') # <-- 修改這一行
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # <-- 修改這一行
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 確保 pose_frames 和 ball_frames 被正確初始化
    pose_frames = {f['frame_idx']: f.get('predictions', []) for f in pose_json.get('frames', [])}
    ball_frames = {frame_idx: box for frame_idx, box in ball_json.get('results', [])}

    prev_center = None
    prev_frame_idx = None
    max_speed_kmh = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- 畫骨架 ---
        # 整合進來的繪圖邏輯
        pose_predictions_for_frame = pose_frames.get(frame_idx, [])
        if pose_predictions_for_frame:
            pitcher_data = pose_predictions_for_frame[0]
            # 直接呼叫本檔案內的繪圖函式
            draw_pitcher_on_frame(frame, pitcher_data)

        # --- 畫棒球 + 計算速度 ---
        if frame_idx in ball_frames:
            current_ball_box = ball_frames[frame_idx]
            # 確保 current_ball_box 不是 None 才能進行 map(int, ...)
            if current_ball_box is not None:
                x1, y1, x2, y2 = map(int, current_ball_box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Baseball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if prev_center is not None and prev_frame_idx is not None:
                    dx = cx - prev_center[0]
                    dy = cy - prev_center[1]
                    distance_pixels = math.sqrt(dx**2 + dy**2)
                    dt = (frame_idx - prev_frame_idx) / fps

                    if dt > 0:
                        distance_m = distance_pixels * pixel_to_meter
                        speed_mps = distance_m / dt
                        speed_kmh = speed_mps * 3.6

                        if min_valid_speed_kmh <= speed_kmh <= max_valid_speed_kmh:
                            max_speed_kmh = max(max_speed_kmh, speed_kmh)

                prev_center = (cx, cy)
                prev_frame_idx = frame_idx
            # 如果 current_ball_box 是 None，則跳過棒球框繪製和速度計算

        # --- 畫最大球速 ---
        label = f"Max Speed: {max_speed_kmh:.1f} km/h"
        cv2.rectangle(frame, (30, 30), (360, 80), (0, 0, 0), -1)  # 黑底
        cv2.putText(frame, label, (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 白字

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_video_path, float(np.round(max_speed_kmh,2))

def save_specific_frames(input_video_path: str, frame_indices: dict) -> dict:
    """
    從影片中儲存特定影格為圖片檔案。
    Args:
        input_video_path: 輸入影片的路徑。
        frame_indices: 包含要儲存的影格名稱和影格編號的字典，例如
                       {"release": 100, "landing": 50, "shoulder": 70}。
                       如果影格編號為 None，則該影格不會被儲存。
    Returns:
        一個字典，包含儲存的圖片路徑，例如
        {"release_frame_path": "path/to/release.jpg", ...}。
    """
    saved_image_paths = {}
    output_dir = "temp_rendered_videos" # 使用相同的暫存目錄
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"無法開啟影片：{input_video_path}")
        return saved_image_paths

    # 預先找出所有需要儲存的影格編號，並排序，以便一次性遍歷影片
    frames_to_save = sorted([
        (name, idx) for name, idx in frame_indices.items() if idx is not None
    ], key=lambda x: x[1])

    current_frame_idx = 0
    frame_pointer = 0 # 用於追蹤 frames_to_save 的進度

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_pointer < len(frames_to_save):
            name, target_frame_idx = frames_to_save[frame_pointer]

            if current_frame_idx == target_frame_idx:
                # 構建圖片儲存路徑
                image_filename = f"{name}_{os.path.basename(input_video_path).replace('.', '_')}.jpg"
                image_path = os.path.join(output_dir, image_filename)
                
                cv2.imwrite(image_path, frame)
                saved_image_paths[f"{name}_frame_path"] = image_path
                print(f"已儲存 {name} 影格至 {image_path}")
                frame_pointer += 1 # 移動到下一個要儲存的影格

        current_frame_idx += 1

        # 如果所有目標影格都已儲存，則提前結束
        if frame_pointer == len(frames_to_save):
            break

    cap.release()
    return saved_image_paths
