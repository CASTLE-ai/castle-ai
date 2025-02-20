import os
import pytest
import numpy as np
import tqdm

# 從你的程式模組引入需要測試的函數
from castle.ui.extract_ui import extract_roi_crop_video, extract_roi_latent, extract_rotation_latent, setting_preprocess

@pytest.fixture
def actual_data_paths():
    """
    根據實際資料修改 storage_path 與 project_name
    例如：假設測試資料放在 /path/to/actual/data/actual_project 下
    """
    storage_path = "tests/"       # <-- 修改為你的實際資料根目錄
    project_name = "test-c_elegans-1min"             # <-- 修改為你的專案資料夾名稱
    # 若路徑不存在，則跳過整個測試
    if not os.path.exists(os.path.join(storage_path, project_name)):
        pytest.skip("實際資料路徑不存在，跳過測試")
    return storage_path, project_name

def test_extract_roi_crop_video_frame_count(actual_data_paths):
    """
    使用實際資料測試 extract_roi_crop_video 功能：
      - 從 config.json 的 source 中取得影片名稱 (此處假設為 "actual_video.mp4")
      - 呼叫 setting_preprocess 初始化 preprocess 物件
      - 呼叫 extract_roi_crop_video 產生 ROI crop 影片
      - 使用 ReadArray 讀取原始影片與輸出影片，確認 frame 數量一致
    """
    storage_path, project_name = actual_data_paths

    select_model = "dinov2_vitb14_reg"  # 請依照你的實際模型設定
    select_roi = "1"                    # 測試時使用 ROI 1
    select_video = "test-c_elegans-1min.mp4"   # 必須與 config.json 中的影片名稱一致
    batch_size = 4                   # 批次處理數，可依實際情況調整

    # 使用 setting_preprocess 初始化 preprocess 物件（此處不啟用中心 ROI 與旋轉）
    preprocess, _ = setting_preprocess(
        storage_path, project_name, select_video,
        center_roi_switch="True",
        center_roi_id=1,
        center_roi_crop_width=210,
        center_roi_crop_height=210,
        rotate_roi_tail_switch="False",
        rotate_roi_tail_id=2
    )

    # 呼叫 extract_roi_crop_video 進行 ROI crop 影片提取
    out_video_path = extract_roi_crop_video(
        storage_path, project_name,
        select_model,
        select_roi,
        select_video,
        batch_size,
        preprocess
    )

    # 驗證輸出影片檔案存在
    assert os.path.exists(out_video_path), f"輸出影片檔案 {out_video_path} 不存在"

    # 使用 ReadArray 讀取原始影片與輸出影片，並比較 frame 數量
    from castle.utils.video_io import ReadArray

    original_video_path = os.path.join(storage_path, project_name, "sources", select_video)
    original_video = ReadArray(original_video_path)
    output_video = ReadArray(out_video_path)

    original_frame_count = len(original_video)
    output_frame_count = len(output_video)

    assert output_frame_count == original_frame_count, (
        f"輸出影片 frame 數量 ({output_frame_count}) 與原始影片 ({original_frame_count}) 不一致"
    )



def test_extract_roi_latent_frame_count(actual_data_paths):
    """
    測試 extract_roi_latent：
      1. 使用 setting_preprocess 初始化 preprocess 物件。
      2. 呼叫 extract_roi_latent，並產生 latent 檔案。
      3. 讀取原始影片 frame 數量，並確認 latent 的 frame 數量與之相同。
    """
    storage_path, project_name = actual_data_paths
    select_model = "dinov2_vitb14_reg"  # 請依照你的實際模型設定
    select_roi = "1"                    # 測試時使用 ROI 1
    select_video = "test-c_elegans-1min.mp4"   # 必須與 config.json 中的影片名稱一致
    batch_size = 4                    # 批次處理數，可依實際情況調整

    # 使用 setting_preprocess 初始化 preprocess 物件（此處不啟用中心 ROI 與旋轉）
    preprocess, _ = setting_preprocess(
        storage_path, project_name, select_video,
        center_roi_switch="True",
        center_roi_id=1,
        center_roi_crop_width=210,
        center_roi_crop_height=210,
        rotate_roi_tail_switch="False",
        rotate_roi_tail_id=2
    )

    latent_file_list = extract_roi_latent(
        storage_path, project_name,
        select_model,
        select_roi,
        select_video,
        batch_size,
        preprocess,
        tqdm
    )
    # 應該會產生至少一個 latent 檔案
    assert latent_file_list, "未產生 ROI latent 檔案"
    latent_file = latent_file_list[0]
    latent_data = np.load(latent_file)
    latent_array = latent_data["latent"]

    # 讀取原始影片的 frame 數量
    from castle.utils.video_io import ReadArray
    original_video_path = os.path.join(storage_path, project_name, "sources", select_video)
    original_video = ReadArray(original_video_path)
    expected_frame_count = len(original_video)

    assert latent_array.shape[0] == expected_frame_count, (
        f"ROI latent frame 數量 {latent_array.shape[0]} 與原始影片 {expected_frame_count} 不一致"
    )

def test_extract_rotation_latent_frame_count(actual_data_paths):
    """
    測試 extract_rotation_latent：
      1. 使用 setting_preprocess 初始化 preprocess 物件。
      2. 呼叫 extract_rotation_latent 產生 rotation latent 檔案。
      3. 每一個原始影片 frame 會產生 24 個 rotation latent（0,15,...,345 度），
         因此預期 latent 的 frame 數量應為原始 frame 數量的 24 倍。
    """
    storage_path, project_name = actual_data_paths
    select_model = "dinov2_vitb14_reg"  # 請依照你的實際模型設定
    select_roi = "1"                    # 測試時使用 ROI 1
    select_video = "test-c_elegans-1min.mp4"   # 必須與 config.json 中的影片名稱一致
    batch_size = 4                    # 批次處理數，可依實際情況調整

    # 使用 setting_preprocess 初始化 preprocess 物件（此處不啟用中心 ROI 與旋轉）
    preprocess, _ = setting_preprocess(
        storage_path, project_name, select_video,
        center_roi_switch="True",
        center_roi_id=1,
        center_roi_crop_width=210,
        center_roi_crop_height=210,
        rotate_roi_tail_switch="False",
        rotate_roi_tail_id=2
    )

    latent_file_list = extract_rotation_latent(
        storage_path, project_name,
        select_model,
        select_roi,
        select_video,
        batch_size,
        preprocess,
        tqdm,
    )
    assert latent_file_list, "未產生 rotation latent 檔案"
    latent_file = latent_file_list[0]
    latent_data = np.load(latent_file)
    latent_array = latent_data["latent"]

    from castle.utils.video_io import ReadArray
    original_video_path = os.path.join(storage_path, project_name, "sources", select_video)
    original_video = ReadArray(original_video_path)
    expected_frame_count = len(original_video) * 24  # 0~345度，每15度一個，共24個

    assert latent_array.shape[0] == expected_frame_count, (
        f"Rotation latent frame 數量 {latent_array.shape[0]} 與預期 {expected_frame_count} 不一致"
    )
