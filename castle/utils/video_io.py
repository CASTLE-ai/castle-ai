"""
影片輸入/輸出工具模組

該模組提供了影片讀取、寫入和字幕生成的功能。
相較於使用 OpenCV，新版本使用 av 庫提供更高效的影片處理能力。

主要功能：
- 高效的影片讀取和隨機存取
- 影片編碼和寫入
- 字幕檔案生成 (SRT, WebVTT)
- 自動資源管理和錯誤處理
"""

import av
import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Tuple, Generator, Union
import logging
from dataclasses import dataclass
from fractions import Fraction

# 設定日誌記錄器
logger = logging.getLogger(__name__)

class VideoIOError(Exception):
    """影片 I/O 特定的錯誤類別"""
    pass


@dataclass
class VideoInfo:
    """
    影片資訊資料類別
    
    用於儲存影片的基本屬性資訊，包括路徑、幀率、影格數量、尺寸和時長等。
    """
    path: Path          # 影片檔案路徑
    fps: float          # 幀率 (frames per second)
    frame_count: int    # 總影格數量
    width: int          # 影片寬度 (像素)
    height: int         # 影片高度 (像素)
    duration: float     # 影片時長 (秒)
    
class VideoIO:
    """
    影片輸入/輸出處理器
    
    提供靜態方法來處理影片的讀取、寫入和基本操作。
    使用 av 庫實現高效的影片處理功能。
    """
    
    @staticmethod
    def load_video(video_path: Union[str, Path]) -> 'VideoReader':
        """
        載入影片檔案
        
        Args:
            video_path: 影片檔案路徑
            
        Returns:
            VideoReader 物件
            
        Raises:
            VideoIOError: 當影片載入失敗時
        """
        try:
            return VideoReader(video_path)
        except Exception as e:
            raise VideoIOError(f"載入影片失敗: {e}")
    
    @staticmethod
    def save_video(
        frames: List[np.ndarray],
        output_path: Union[str, Path],
        fps: float = 30.0,
        crf: int = 15,
        codec: str = 'libx264'
    ) -> None:
        """
        儲存影片
        
        Args:
            frames: 影格陣列列表
            output_path: 輸出檔案路徑
            fps: 幀率
            crf: 壓縮品質 (0-51, 數值越小品質越好)
            codec: 編碼器名稱
            
        Raises:
            VideoIOError: 當影片儲存失敗時
            ValueError: 當輸入資料無效時
        """
        if not frames:
            raise ValueError("沒有影格資料可儲存")
        
        if not all(isinstance(frame, np.ndarray) for frame in frames):
            raise ValueError("所有影格必須是 numpy 陣列")
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用 VideoWriter 類來寫入影片
            with VideoWriter(str(output_path), fps, crf, codec) as writer:
                for frame in frames:
                    writer.write_frame(frame)
            
            logger.info(f"成功儲存影片到 {output_path}")
            
        except Exception as e:
            raise VideoIOError(f"儲存影片失敗: {e}")
        
    @staticmethod
    def get_frame(video: 'VideoReader', frame_idx: int) -> np.ndarray:
        """
        從 VideoReader 獲取指定影格
        
        Args:
            video: VideoReader 實例
            frame_idx: 影格索引
            
        Returns:
            影格陣列 (RGB)
        """
        return video.get_frame(frame_idx)
        
class VideoReader:
    """
    影片讀取器
    
    使用 av 庫實現高效的影片讀取功能，支援隨機存取和批次讀取。
    相較於傳統的 OpenCV 方法，提供更好的效能和準確性。
    
    主要功能：
    - 高效的隨機影格存取
    - 自動資源管理
    - 影格快取機制
    - 批次讀取支援
    
    範例用法：
        # 基本使用
        reader = VideoReader("video.mp4")
        frame = reader.get_frame(100)  # 獲取第 100 幀
        
        # 作為 context manager 使用
        with VideoReader("video.mp4") as reader:
            for i, frame in reader.iterate_frames(0, 100, 5):
                process_frame(frame)
    """
    
    def __init__(self, video_path: Union[str, Path]):
        """
        初始化影片讀取器
        
        Args:
            video_path: 影片檔案路徑
            
        Raises:
            VideoIOError: 當影片初始化失敗時
            FileNotFoundError: 當影片檔案不存在時
        """
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"影片檔案不存在: {video_path}")
        
        
        try:
            # 開啟影片容器
            self.container = av.open(str(self.path))
            self.video_stream = self.container.streams.video[0]
            
            # 計算影格索引轉換參數
            self.fps = float(self.video_stream.average_rate)
            time_base = self.video_stream.time_base
            self.pts2index = time_base * self.video_stream.average_rate
            
            # 獲取影片尺寸
            self.width = self.video_stream.width
            self.height = self.video_stream.height
            
            # 計算總影格數（需要特殊處理以確保準確性）
            self.frame_count = self._calculate_frame_count()
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0
            
            # 內部狀態
            self._current_index = -1
            self._frame_cache = {}
            self._closed = False
            
            logger.debug(f"影片讀取器初始化完成: {self.path}")
            logger.debug(f"影片資訊: {self.width}x{self.height}, {self.fps:.2f}fps, {self.frame_count} frames")
            
        except Exception as e:
            raise VideoIOError(f"初始化影片讀取器失敗: {e}")
        
        # for old version
        self.video_path = self.path
        self.video_name = os.path.basename(self.video_path)
        self.total_frames = self.__len__()
    
    def __enter__(self) -> 'VideoReader':
        """Context manager 進入"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager 退出"""
        self.close()
    
    def __len__(self) -> int:
        """返回總影格數量"""
        return self.frame_count
    
    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """支援 frame = reader[index] 語法"""
        return self.get_frame(frame_idx)
    
    def __del__(self) -> None:
        """解構函數，確保資源被正確釋放"""
        self.close()
    
    def _calculate_frame_count(self) -> int:
        """
        計算總影格數量
        
        直接使用 stream.frames，避免在初始化時讀取影格導致容器狀態混亂。
        
        Returns:
            總影格數量
        """
        # 首先嘗試使用 stream 的 frames 屬性
        stream_frames = self.video_stream.frames
        
        if stream_frames and stream_frames > 0:
            # 直接信任 stream.frames，不在初始化時讀取影格
            logger.debug(f"使用 stream.frames: {stream_frames}")
            return stream_frames
        
        # 如果 stream.frames 不可用，使用二分搜尋法
        logger.warning(f"stream.frames 不可用 ({stream_frames})，使用二分搜尋法計算影格數")
        return self._binary_search_frame_count()
    
    def _binary_search_frame_count(self) -> int:
        """使用二分搜尋法確定影格數量"""
        return self._binary_search_frame_count_from(0)
    
    def _binary_search_frame_count_from(self, start_frame: int) -> int:
        """從指定影格開始使用二分搜尋法確定影格數量"""
        low, high = start_frame, 1000000  # 假設最大 1m 幀
        
        while low < high:
            mid = (low + high + 1) // 2
            try:
                self._get_frame_direct(mid)
                low = mid
            except (RuntimeError, av.error.EOFError, IndexError, Exception):
                # 任何讀取錯誤都表示該影格不存在
                high = mid - 1
        
        return low + 1
    
    def _get_frame_direct(self, frame_index: int) -> np.ndarray:
        """
        直接讀取影格（內部方法）
        
        Args:
            frame_index: 影格索引
            
        Returns:
            影格陣列 (RGB)
            
        Raises:
            RuntimeError: 當讀取失敗時
        """
        if self._closed:
            raise RuntimeError("影片讀取器已關閉")
        
        # 檢索是否為順序讀取（但排除第0幀，第0幀總是需要 seek）
        if frame_index > 0 and frame_index == self._current_index + 1:
            try:
                self._current_index = frame_index
                frame = next(self.container.decode(self.video_stream))
                return frame.to_rgb().to_ndarray()
            except (StopIteration, av.error.EOFError):
                # 順序讀取失敗（到達檔案結尾或其他原因），改用 seek 方式
                logger.debug(f"順序讀取影格 {frame_index} 失敗，改用 seek 方式")
                pass
            except Exception as e:
                # 其他異常也回退到 seek 方式
                logger.debug(f"順序讀取影格 {frame_index} 時發生異常: {e}，改用 seek 方式")
                pass
        
        # 非順序讀取、第0幀或順序讀取失敗時，使用 seek 方式
        return self._seek_and_read_frame(frame_index)
    
    def _seek_and_read_frame(self, frame_index: int) -> np.ndarray:
        """
        使用 seek 方式讀取指定影格
        
        Args:
            frame_index: 影格索引
            
        Returns:
            影格陣列 (RGB)
            
        Raises:
            RuntimeError: 當讀取失敗時
        """
        try:
            # 計算時間戳並 seek
            timestamp = frame_index / self.pts2index
            self.container.seek(int(timestamp), stream=self.video_stream, backward=True)
            
            # 尋找匹配的影格
            target_frame = None
            frames_checked = 0
            max_frames_to_check = 10  # 限制檢查的影格數量，避免無限循環
            
            try:
                for frame in self.container.decode(self.video_stream):
                    frames_checked += 1
                    index = int(frame.pts * self.pts2index)
                    
                    if index == frame_index:
                        target_frame = frame
                        break
                    elif index > frame_index:
                        # 已經超過目標影格，停止搜尋
                        break
                    elif frames_checked >= max_frames_to_check:
                        # 避免無限循環
                        break
            except (av.error.EOFError, StopIteration):
                # 到達檔案結尾，停止搜尋
                logger.debug(f"在 seek 讀取過程中到達檔案結尾，影格 {frame_index}")
                pass
            
            if target_frame is not None:
                self._current_index = frame_index
                return target_frame.to_rgb().to_ndarray()
            
            # 如果精確匹配失敗，嘗試容錯方式：取最接近的影格
            return self._fallback_frame_read(frame_index)
            
        except Exception as e:
            raise RuntimeError(f"使用 seek 方式讀取影格 {frame_index} 失敗: {e}")
    
    def _fallback_frame_read(self, frame_index: int) -> np.ndarray:
        """
        容錯影格讀取：當精確匹配失敗時，嘗試讀取最接近的影格
        
        Args:
            frame_index: 影格索引
            
        Returns:
            影格陣列 (RGB)
            
        Raises:
            RuntimeError: 當讀取失敗時
        """
        try:
            # 重新 seek 並尋找最接近的影格
            timestamp = frame_index / self.pts2index
            self.container.seek(int(timestamp), stream=self.video_stream, backward=True)
            
            closest_frame = None
            min_distance = float('inf')
            frames_checked = 0
            max_frames_to_check = 5
            
            try:
                for frame in self.container.decode(self.video_stream):
                    frames_checked += 1
                    index = int(frame.pts * self.pts2index)
                    distance = abs(index - frame_index)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_frame = frame
                    
                    if frames_checked >= max_frames_to_check or index > frame_index + 2:
                        break
            except (av.error.EOFError, StopIteration):
                # 到達檔案結尾，停止搜尋
                logger.debug(f"在容錯讀取過程中到達檔案結尾，影格 {frame_index}")
                pass
            
            if closest_frame is not None:
                self._current_index = frame_index
                logger.debug(f"使用容錯模式讀取影格 {frame_index}，實際距離: {min_distance}")
                return closest_frame.to_rgb().to_ndarray()
            
            raise RuntimeError(f"無法找到影格 {frame_index} 的任何可用替代")
            
        except Exception as e:
            raise RuntimeError(f"容錯讀取影格 {frame_index} 失敗: {e}")
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        獲取指定影格
        
        Args:
            frame_idx: 影格索引
            
        Returns:
            影格陣列 (RGB 格式)
            
        Raises:
            IndexError: 當影格索引超出範圍時
            RuntimeError: 當讀取失敗時
        """
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(f"影格索引 {frame_idx} 超出範圍 [0, {self.frame_count})")
        
        # 檢查快取
        if frame_idx in self._frame_cache:
            return self._frame_cache[frame_idx]
        
        # 讀取影格
        try:
            frame = self._get_frame_direct(frame_idx)
        except RuntimeError as e:
            # 如果讀取失敗且索引接近邊界，可能是 frame_count 計算不準確
            if frame_idx >= self.frame_count - 10:
                logger.warning(f"影格 {frame_idx} 讀取失敗，可能超出實際影片範圍: {e}")
                raise IndexError(f"影格索引 {frame_idx} 可能超出實際影片範圍")
            else:
                # 其他錯誤直接拋出
                raise
        
        # 快取管理（限制快取大小）
        if len(self._frame_cache) >= 100:
            # 移除最舊的快取項目
            oldest_key = next(iter(self._frame_cache))
            del self._frame_cache[oldest_key]
        
        self._frame_cache[frame_idx] = frame
        logger.debug(f"成功讀取並快取影格 {frame_idx}")
        
        return frame
    
    def iterate_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        迭代影片影格
        
        Args:
            start: 起始影格索引
            end: 結束影格索引（不包含），None 表示到影片結尾
            step: 步長
            
        Yields:
            (影格索引, 影格陣列) 元組
            
        Raises:
            ValueError: 當參數無效時
        """
        if end is None:
            end = self.frame_count
        
        if start < 0 or start >= self.frame_count:
            raise ValueError(f"起始索引 {start} 無效")
        
        if end > self.frame_count:
            end = self.frame_count
        
        if step <= 0:
            raise ValueError("步長必須為正數")
        
        for i in range(start, end, step):
            try:
                frame = self.get_frame(i)
                yield i, frame
            except Exception as e:
                logger.warning(f"跳過影格 {i}: {e}")
                continue
    
    def get_batch_frames(self, indices: List[int]) -> List[np.ndarray]:
        """
        批次獲取影格
        
        Args:
            indices: 影格索引列表
            
        Returns:
            影格陣列列表
            
        Raises:
            ValueError: 當索引列表為空時
        """
        if not indices:
            raise ValueError("索引列表不能為空")
        
        frames = []
        for idx in indices:
            try:
                frame = self.get_frame(idx)
                frames.append(frame)
            except Exception as e:
                logger.warning(f"跳過影格 {idx}: {e}")
                # 添加空影格佔位符（與原索引對應）
                frames.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))
        
        return frames
    
    def get_info(self) -> VideoInfo:
        """
        獲取影片資訊
        
        Returns:
            VideoInfo 實例，包含影片的基本資訊
        """
        return VideoInfo(
            path=self.path,
            fps=self.fps,
            frame_count=self.frame_count,
            width=self.width,
            height=self.height,
            duration=self.duration
        )
    
    def clear_cache(self) -> None:
        """清除影格快取"""
        self._frame_cache.clear()
        logger.debug("影格快取已清除")
    
    def close(self) -> None:
        """關閉影片讀取器並釋放資源"""
        if not self._closed:
            try:
                if hasattr(self, 'container') and self.container:
                    self.container.close()
                self._frame_cache.clear()
                self._closed = True
                logger.debug("影片讀取器已關閉")
            except Exception as e:
                logger.error(f"關閉影片讀取器時發生錯誤: {e}")


class VideoWriter:
    """
    影片寫入器
    
    使用 av 庫實現高效的影片編碼和寫入功能。
    基於提供的 WriteArray 範例實現，支援多種編碼器和品質設定。
    
    範例用法：
        # 基本使用
        writer = VideoWriter("output.mp4", fps=30.0)
        for frame in frames:
            writer.write_frame(frame)
        writer.close()
        
        # 作為 context manager 使用
        with VideoWriter("output.mp4", fps=30.0, crf=20) as writer:
            for frame in frames:
                writer.write_frame(frame)
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float = 30.0,
        crf: int = 15,
        codec: str = 'libx264'
    ):
        """
        初始化影片寫入器
        
        Args:
            output_path: 輸出檔案路徑
            fps: 幀率
            crf: 壓縮品質 (0-51, 數值越小品質越好，檔案越大)
            codec: 編碼器名稱
            
        Raises:
            VideoIOError: 當初始化失敗時
        """
        try:
            self.output_path = Path(output_path)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.output = av.open(str(self.output_path), 'w')
            # 轉換 fps 為 Fraction 對象以支援 PyAV 15.1.0
            fps_fraction = Fraction(fps).limit_denominator()
            self.stream = self.output.add_stream(codec, rate=fps_fraction)
            self.stream.options = {'crf': str(crf)}
            self.stream.pix_fmt = 'yuv420p'
            
            self._initialized = False
            self._closed = False
            self._frame_count = 0
            
            logger.debug(f"影片寫入器初始化完成: {self.output_path}")
            
        except Exception as e:
            raise VideoIOError(f"初始化影片寫入器失敗: {e}")
        
        # for old version
        self.video_path = output_path
        self.video_name = os.path.basename(self.video_path)
        
    
    def __enter__(self) -> 'VideoWriter':
        """Context manager 進入"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager 退出"""
        self.close()
    
    def __del__(self) -> None:
        """解構函數，確保資源被正確釋放"""
        self.close()
    
    def write_frame(self, frame: np.ndarray) -> None:
        """
        寫入一個影格
        
        Args:
            frame: 影格陣列，形狀為 (H, W, 3)，RGB 格式
            
        Raises:
            ValueError: 當影格格式無效時
            VideoIOError: 當寫入失敗時
        """
        if self._closed:
            raise VideoIOError("影片寫入器已關閉")
        
        if not isinstance(frame, np.ndarray):
            raise ValueError("影格必須是 numpy 陣列")
        
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("預期影格形狀為 (H, W, 3)")
        
        try:
            # 首次寫入時設定影片尺寸
            if not self._initialized:
                self.stream.height, self.stream.width = frame.shape[:2]
                self._initialized = True
                logger.debug(f"設定影片尺寸: {self.stream.width}x{self.stream.height}")
            
            # 確保影格格式正確
            if frame.dtype != np.uint8:
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # 建立 VideoFrame 並編碼
            av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            
            for packet in self.stream.encode(av_frame):
                self.output.mux(packet)
            
            self._frame_count += 1
            
            if self._frame_count % 100 == 0:
                logger.debug(f"已寫入 {self._frame_count} 個影格")
        
        except Exception as e:
            raise VideoIOError(f"寫入影格失敗: {e}")
    
    def append(self, frame: np.ndarray) -> None:
        """
        追加影格（向後相容性方法）
        
        這是為了與舊版本 WriteArray 相容而提供的方法。
        內部調用 write_frame 方法。
        
        Args:
            frame: 影格陣列，形狀為 (H, W, 3)，RGB 格式
            
        Raises:
            ValueError: 當影格格式無效時
            VideoIOError: 當寫入失敗時
        """
        self.write_frame(frame)
    
    def close(self) -> None:
        """關閉影片寫入器並完成編碼"""
        if self._closed:
            return
        
        try:
            # 刷新編碼器緩衝區
            if hasattr(self, 'stream'):
                for packet in self.stream.encode():
                    self.output.mux(packet)
            
            # 關閉輸出檔案
            if hasattr(self, 'output'):
                self.output.close()
            
            self._closed = True
            logger.info(f"影片寫入完成: {self.output_path}, 總共 {self._frame_count} 個影格")
            
        except Exception as e:
            logger.error(f"關閉影片寫入器時發生錯誤: {e}")
        
class SubtitleGenerator:
    """
    字幕生成器
    
    提供字幕檔案的建立和儲存功能，支援多種字幕格式。
    可用於為影片自動生成時間同步的字幕檔案。
    
    支援格式：
    - SRT (SubRip Text)
    - WebVTT (Web Video Text Tracks)
    
    範例用法：
        generator = SubtitleGenerator()
        generator.add_subtitle(0.0, 5.0, "第一段字幕")
        generator.add_subtitle(5.0, 10.0, "第二段字幕")
        generator.save("output.srt", format="srt")
    """
    
    def __init__(self):
        """初始化字幕生成器"""
        self.subtitles = []
        logger.debug("字幕生成器初始化完成")
        
    def add_subtitle(
        self,
        start_time: float,
        end_time: float,
        text: str
    ) -> None:
        """
        添加字幕項目
        
        Args:
            start_time: 開始時間（秒）
            end_time: 結束時間（秒）
            text: 字幕文字內容
            
        Raises:
            ValueError: 當時間參數無效時
        """
        if start_time < 0 or end_time < 0:
            raise ValueError("時間不能為負數")
        
        if start_time >= end_time:
            raise ValueError("開始時間必須小於結束時間")
        
        if not text.strip():
            raise ValueError("字幕文字不能為空")
        
        self.subtitles.append({
            'start': start_time,
            'end': end_time,
            'text': text.strip()
        })
        
        logger.debug(f"添加字幕: {start_time:.1f}s-{end_time:.1f}s: {text[:30]}...")
        
    def save(self, output_path: Union[str, Path], format: str = 'srt') -> None:
        """
        儲存字幕檔案
        
        Args:
            output_path: 輸出檔案路徑
            format: 字幕格式 ('srt', 'vtt')
            
        Raises:
            ValueError: 當格式不支援時
            IOError: 當檔案寫入失敗時
        """
        if not self.subtitles:
            logger.warning("沒有字幕資料可儲存")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == 'srt':
                self._save_srt(output_path)
            elif format.lower() == 'vtt':
                self._save_vtt(output_path)
            else:
                raise ValueError(f"不支援的字幕格式: {format}")
            
            logger.info(f"字幕檔案儲存完成: {output_path} ({len(self.subtitles)} 個項目)")
            
        except Exception as e:
            raise IOError(f"儲存字幕檔案失敗: {e}")
            
    def _save_srt(self, output_path: Path) -> None:
        """儲存為 SRT 格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sub in enumerate(self.subtitles, 1):
                f.write(f"{i}\n")
                f.write(f"{self._format_srt_time(sub['start'])} --> ")
                f.write(f"{self._format_srt_time(sub['end'])}\n")
                f.write(f"{sub['text']}\n\n")
                
    def _save_vtt(self, output_path: Path) -> None:
        """儲存為 WebVTT 格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for sub in self.subtitles:
                f.write(f"{self._format_vtt_time(sub['start'])} --> ")
                f.write(f"{self._format_vtt_time(sub['end'])}\n")
                f.write(f"{sub['text']}\n\n")
                
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """
        格式化 SRT 時間格式
        
        Args:
            seconds: 時間（秒）
            
        Returns:
            SRT 格式時間字串 (HH:MM:SS,mmm)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """
        格式化 WebVTT 時間格式
        
        Args:
            seconds: 時間（秒）
            
        Returns:
            WebVTT 格式時間字串 (HH:MM:SS.mmm)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def clear(self) -> None:
        """清除所有字幕項目"""
        self.subtitles.clear()
        logger.debug("字幕項目已清除")
    
    def get_subtitle_count(self) -> int:
        """獲取字幕項目數量"""
        return len(self.subtitles)
    
    def get_total_duration(self) -> float:
        """
        獲取字幕總時長
        
        Returns:
            總時長（秒），如果沒有字幕則返回 0
        """
        if not self.subtitles:
            return 0.0
        
        return max(sub['end'] for sub in self.subtitles)
    

ReadArray = VideoReader
WriteArray = VideoWriter