# 小檸檬版本
import os
import json
import numpy as np
import cv2
# landing.py
"""
落地那一幀
"""
def detect_landing_frame(pose_sequence, release_frame, back_offset=9):
    """
    根據 release_frame 向前推 back_offset 幀作為落地點
    - pose_sequence: 骨架序列（list of dict，含 frame 與 keypoints）
    - release_frame: 偵測出的出手幀編號
    - back_offset: 預設往前 9 幀
    """
    candidate_index = next((i for i, item in enumerate(pose_sequence) if item["frame"] == release_frame), None)
    if candidate_index is None:
        print(f"❌ 找不到 release_frame = {release_frame} 的對應資料")
        return None

    landing_index = candidate_index - back_offset
    if landing_index < 0:
        print(f"❌ 推估 index = {landing_index} 超出範圍")
        return None

    landing_item = pose_sequence[landing_index]
    landing_frame = landing_item["frame"]

    return landing_frame

# release.py
"""
出手那一幀
"""
def detect_release_frame(pose_sequence):
    candidate_frames = []

    for item in pose_sequence:
        frame_idx = item["frame"]
        keypoints = item["keypoints"]
        if keypoints.shape[0] < 17:
            continue

        rs = COCO_KEYPOINTS["right_shoulder"]
        re = COCO_KEYPOINTS["right_elbow"]
        rw = COCO_KEYPOINTS["right_wrist"]

        right_shoulder = keypoints[rs]
        right_elbow = keypoints[re]
        right_wrist = keypoints[rw]

        # 1. 手腕高於肩膀（Y 軸）
        wrist_above_shoulder = right_wrist[1] < right_shoulder[1]
        # 2. 手肘在手腕後方（X 軸）
        elbow_behind_wrist = right_elbow[0] < right_wrist[0]
        # 3. 肘角
        elbow_angle = calculate_pixel_angle_from_points(
            right_wrist[:2], right_elbow[:2], right_shoulder[:2]
        )
        # 4. 手臂長度（wrist→elbow + elbow→shoulder）
        arm_length = np.linalg.norm(right_wrist - right_elbow) + np.linalg.norm(right_elbow - right_shoulder)

        if wrist_above_shoulder and elbow_behind_wrist:
            candidate_frames.append({
                "frame": frame_idx,
                "elbow_angle": elbow_angle,
                "arm_length": arm_length
            })

    if not candidate_frames:
        print("⚠️ 沒有符合條件的出手幀")
        return None

    # 🔍 先挑肘角最大，再用臂長決勝負（肘角差距容忍 5 度內）
    max_angle = max(f["elbow_angle"] for f in candidate_frames)
    top_angle_candidates = [f for f in candidate_frames if abs(f["elbow_angle"] - max_angle) < 5]
    best_frame = max(top_angle_candidates, key=lambda f: f["arm_length"])

    return best_frame["frame"]

# shoulder.py
"""
肩膀最大那一幀
"""
def detect_shoulder_frame(pose_sequence, release_frame):
    """
    偵測肩膀最開啟的幀：
    - 起始：右手腕高於右肩（代表投球已啟動）
    - 條件：右手腕仍在右肩左側且未落下
    - 評分：肩寬前 3 名 → 選最大肩角

    return: shoulder_frame 編號（int）或 None
    """
    candidate_list = []
    start_found = False

    LEFT_SHOULDER = COCO_KEYPOINTS["left_shoulder"]
    RIGHT_SHOULDER = COCO_KEYPOINTS["right_shoulder"]
    LEFT_HIP = COCO_KEYPOINTS["left_hip"]
    RIGHT_WRIST = COCO_KEYPOINTS["right_wrist"]

    for item in pose_sequence:
        frame_idx = item["frame"]
        if frame_idx > release_frame:
            break

        keypoints = item["keypoints"]
        if np.min(keypoints[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_WRIST], 2]) < 0.3:
            continue

        l_sh = keypoints[LEFT_SHOULDER][:2]
        r_sh = keypoints[RIGHT_SHOULDER][:2]
        l_hip = keypoints[LEFT_HIP][:2]
        r_wr = keypoints[RIGHT_WRIST][:2]

        # 起始條件：右手腕高於右肩
        if not start_found:
            if r_wr[1] < r_sh[1]:
                start_found = True
            else:
                continue

        # 排除：右手腕落下 或 手腕已超過肩膀
        if r_wr[0] > r_sh[0] or r_wr[1] >= r_sh[1]:
            continue

        # 肩膀開啟角度（l_sh - r_sh - l_hip）
        angle = calculate_pixel_angle_from_points(l_sh, r_sh, l_hip)
        shoulder_distance = abs(r_sh[0] - l_sh[0])
        candidate_list.append((angle, shoulder_distance, frame_idx))

    if not candidate_list:
        print("❌ 無法找到符合條件的肩膀開啟幀")
        return None

    # 取肩膀 X 軸距離最大的前三名 → 選角度最大者
    top3 = sorted(candidate_list, key=lambda x: -x[1])[:3]
    best = max(top3, key=lambda x: x[0])
    shoulder_frame = best[2]

    return shoulder_frame

"""
取特徵
"""
COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def feature2kinematic(pose_sequence, release_frame, landing_frame, shoulder_frame=None):
    """
    從姿勢序列與關鍵幀中提取基本 2D 力學特徵

    輸入：
        pose_sequence: list of {"frame": int, "keypoints": np.ndarray(17, 3)}
        release_frame: 出手幀編號
        landing_frame: 踏地幀編號
        shoulder_frame: 肩膀展開幀（暫未使用，可預留）

    回傳：
        dict 包含 6 項特徵（英文變數命名）：
        - Trunk_flexion_excursion
        - Pelvis_obliquity_at_FC
        - Trunk_rotation_at_BR
        - Shoulder_abduction_at_BR
        - Trunk_flexion_at_BR
        - Trunk_lateral_flexion_at_HS
    """
    kinematic = {}

    # === Trunk flexion excursion（軀幹前彎動作幅度）===
    # 透過每幀的肩膀中心 Y 與髖部中心 Y 的差值，計算最大與最小值差，代表整體前傾變化範圍
    trunk_flexions = []
    for item in pose_sequence:
        keypoints = item["keypoints"]
        ls, rs = COCO_KEYPOINTS["left_shoulder"], COCO_KEYPOINTS["right_shoulder"]
        lh, rh = COCO_KEYPOINTS["left_hip"], COCO_KEYPOINTS["right_hip"]
        shoulder_y = (keypoints[ls][1] + keypoints[rs][1]) / 2
        hip_y = (keypoints[lh][1] + keypoints[rh][1]) / 2
        trunk_flexions.append(shoulder_y - hip_y)
    kinematic["Trunk_flexion_excursion"] = max(trunk_flexions) - min(trunk_flexions)

    # === Pelvis obliquity at FC（踏地瞬間的骨盆傾斜角）===
    # 用左髖與右髖的 Y 值差代表骨盆左右傾斜，Y 差越大表示傾斜越明顯
    keypoints_fc = get_keypoints_at(pose_sequence, landing_frame)
    if keypoints_fc is not None:
        lh, rh = COCO_KEYPOINTS["left_hip"], COCO_KEYPOINTS["right_hip"]
        pelvis_obliquity = keypoints_fc[lh][1] - keypoints_fc[rh][1]
        kinematic["Pelvis_obliquity_at_FC"] = pelvis_obliquity

    # === Trunk rotation at BR（釋球瞬間的軀幹旋轉角）===
    # 取左右肩的 X 向量差並轉為角度，表示橫向旋轉程度（水平旋轉角）
    keypoints_br = get_keypoints_at(pose_sequence, release_frame)
    if keypoints_br is not None:
        ls, rs = COCO_KEYPOINTS["left_shoulder"], COCO_KEYPOINTS["right_shoulder"]
        shoulder_vec = keypoints_br[ls][:2] - keypoints_br[rs][:2]
        trunk_rotation = np.arctan2(shoulder_vec[1], shoulder_vec[0]) * 180 / np.pi
        kinematic["Trunk_rotation_at_BR"] = trunk_rotation

        # === Shoulder abduction at BR（肩部外展角）===
        # 以「右手腕–右手肘–右肩膀」三點形成的角度表示肩膀抬起程度（右投）
        re = COCO_KEYPOINTS["right_elbow"]
        rw = COCO_KEYPOINTS["right_wrist"]
        rs = COCO_KEYPOINTS["right_shoulder"]
        shoulder_abduction = calculate_pixel_angle_from_points(
            keypoints_br[rw][:2], keypoints_br[re][:2], keypoints_br[rs][:2]
        )
        kinematic["Shoulder_abduction_at_BR"] = shoulder_abduction

        # === Trunk flexion at BR（釋球瞬間的軀幹前傾角）===
        # 釋球幀的肩膀中心 Y 與骨盆中心 Y 差值，數值越大表示向前傾越多
        lh, rh = COCO_KEYPOINTS["left_hip"], COCO_KEYPOINTS["right_hip"]
        shoulder_y = (keypoints_br[ls][1] + keypoints_br[rs][1]) / 2
        hip_y = (keypoints_br[lh][1] + keypoints_br[rh][1]) / 2
        kinematic["Trunk_flexion_at_BR"] = shoulder_y - hip_y

    # === Trunk lateral flexion at HS（起投瞬間的軀幹側彎角）===
    # 起投幀左右肩膀的 Y 軸差異，表示是否側向一側（正值：左肩低於右肩）
    keypoints_hs = get_keypoints_at(pose_sequence, pose_sequence[0]["frame"])
    if keypoints_hs is not None:
        ls, rs = COCO_KEYPOINTS["left_shoulder"], COCO_KEYPOINTS["right_shoulder"]
        kinematic["Trunk_lateral_flexion_at_HS"] = keypoints_hs[ls][1] - keypoints_hs[rs][1]

    return kinematic

def extract_pitching_biomechanics(result):
    """
    接收已處理好的 pose_sequence，偵測出手、落地、肩膀展開幀。
    Args:
        pose_sequence: list[dict]，每個元素為 {"frame": int, "keypoints": np.ndarray(17, 3)}

    Returns:
        dict: 包含 release、landing、shoulder 三幀與總長度
    """

    # ✅ 先手動轉成 pose_sequence
    pose_sequence = load_pose_from_response(result)
    
    if not pose_sequence:
        print("❌ pose_sequence 為空")
        return {}

    # === 出手幀 ===
    release_frame = detect_release_frame(pose_sequence)
    if release_frame is None:
        print("❌ 偵測不到出手幀")
        return {}

    # === 落地幀 ===
    landing_frame = detect_landing_frame(pose_sequence, release_frame)

    # === 肩膀最展開幀 ===
    shoulder_frame = detect_shoulder_frame(pose_sequence, release_frame)

    kinematic = feature2kinematic(pose_sequence, release_frame, landing_frame)

    return {
    "release_frame": release_frame,
    "landing_frame": landing_frame,
    "shoulder_frame": shoulder_frame,
    "total_frames": len(pose_sequence),
    "Trunk_flexion_excursion": kinematic.get("Trunk_flexion_excursion"),
    "Pelvis_obliquity_at_FC": kinematic.get("Pelvis_obliquity_at_FC"),
    "Trunk_rotation_at_BR": kinematic.get("Trunk_rotation_at_BR"),
    "Shoulder_abduction_at_BR": kinematic.get("Shoulder_abduction_at_BR"),
    "Trunk_flexion_at_BR": kinematic.get("Trunk_flexion_at_BR"),
    "Trunk_lateral_flexion_at_HS": kinematic.get("Trunk_lateral_flexion_at_HS"),
}


# utils.py
"""
通用函式：讀取 pose_sequence、計算角度等
"""
def load_pose_from_response(result_json):
    """
    從 FastAPI 回傳的 JSON 解析出 pose_sequence 格式
    - result_json: API 回傳的 dict
    - 回傳: list[{"frame": int, "keypoints": np.ndarray(17, 3)}]
    """
    pose_sequence = []

    for frame in result_json["frames"]:
        if not frame["predictions"]:
            continue

        keypoints = np.array(frame["predictions"][0]["keypoints"])  # shape: (17, 2 or 3)

        if keypoints.shape[1] == 2:
            conf = np.ones((17, 1), dtype=np.float32)
            keypoints = np.concatenate([keypoints, conf], axis=-1)

        pose_sequence.append({
            "frame": frame["frame_idx"],
            "keypoints": keypoints
        })

    return pose_sequence


def get_keypoints_at(pose_sequence, frame_id):
    """
    根據幀編號 frame_id 回傳該幀的 keypoints。
    - pose_sequence: list of dict，格式為 [{"frame": int, "keypoints": np.ndarray(17, 3)}]
    - frame_id: int，欲查找的幀號

    回傳：該幀的 keypoints，或 None 若找不到。
    """
    for item in pose_sequence:
        if item["frame"] == frame_id:
            return item["keypoints"]
    return None


def calculate_pixel_angle(a, b, c):
    """
    計算以點 b 為中心，夾在向量 ab 和 cb 之間的夾角（像素座標）
    - a, b, c: np.array([x, y])
    - 回傳: angle in degrees
    """
    ab = a - b
    cb = c - b

    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return None

    cosine_angle = np.dot(ab, cb) / (norm_ab * norm_cb)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def calculate_pixel_angle_from_points(a, b, c):
    """
    舊版本相容函式，輸入為 list 或 tuple（自動轉 np.array）
    """
    return calculate_pixel_angle(np.array(a), np.array(b), np.array(c))

