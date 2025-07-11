from typing import Dict, Tuple

def calculate_score_from_comparison(features: dict, profile_data: dict) -> Tuple[int, Dict]:
    """
    將使用者的生物力學特徵與一個基準模型進行比較。

    此函式根據每個特徵與設定檔資料中的平均值和標準差的 Z-score 來計算投球品質分數。
    它會回傳最終的綜合分數以及每個特徵比較的詳細分解。

    Args:
        features: 包含使用者生物力學特徵的字典。
        profile_data: 包含基準模型統計數據（平均值、標準差等）的字典。

    Returns:
        一個元組，包含：
        - final_score (int): 整體品質分數 (0-100)。
        - comparison_details (Dict): 包含每個特徵詳細分數的字典。
    """
    if not profile_data:
        return 0, {}

    total_score = 0
    feature_count = 0
    comparison_details = {}

    for key, user_value in features.items():
        # 在 profile_data 中尋找對應的鍵 (轉為小寫以確保匹配)
        profile_stats = profile_data.get(key.lower())
        
        # 如果模型中沒有這個特徵或使用者沒有此數值，則跳過
        if not profile_stats or user_value is None:
            continue

        mean = profile_stats.get('mean')
        std = profile_stats.get('std')

        # 如果缺乏平均值或標準差，則無法比較
        if mean is None or std is None:
            continue
            
        # 計算 Z-score，處理標準差為 0 的情況
        if std == 0:
            z_score = 0.0
        else:
            # Z-score 反映了用戶數值偏離平均值的標準差倍數
            z_score = abs((user_value - mean) / std)

        # 分數在 Z-score 為 0 時為 100，隨著 Z-score 增加而線性下降，在 Z-score=4 時降至 0
        feature_score = max(0, 100 - z_score * 25)

        total_score += feature_score
        feature_count += 1
        
        # 儲存每個特徵的詳細比較結果
        comparison_details[key] = {
            "user_value": user_value,
            "mean": mean,
            "std": std,
            "z_score": z_score,
            "score": int(feature_score)
        }

    # 如果沒有任何特徵被成功比較，則回傳 0
    if feature_count == 0:
        return 0, {}

    # 最終分數是所有特徵分數的平均值
    final_score = int(total_score / feature_count)
    
    return final_score, comparison_details