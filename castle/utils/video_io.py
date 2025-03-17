import av
import os

# import decord
# from decord import VideoReader, cpu, gpu
# import os

# class ReadArray:
#     def __init__(self, video_path):
#         self.video_path = video_path
#         self.video_name = os.path.basename(video_path)
#         # 使用 decord 讀取影片，並設定執行環境為 CPU
#         self.vr = VideoReader(video_path, ctx=cpu(0))
#         # self.vr = VideoReader(video_path, ctx=gpu(0))
#         self.fps = self.vr.get_avg_fps()
#         self.total_frames = len(self.vr)
#         self.index = -1  # 初始化目前讀取的影格索引

#     def __len__(self):
#         return self.total_frames

#     def __getitem__(self, frame_index):
#         if frame_index < 0 or frame_index >= self.total_frames:
#             raise IndexError("frame index out of range")
#         self.index = frame_index
#         # 直接使用 decord 隨機存取影格，並轉換成 numpy array
#         frame = self.vr[frame_index]
#         return frame.asnumpy()

#     def __del__(self):
#         # decord 會自動管理資源，不需要手動釋放
#         pass

# # 測試範例
# if __name__ == '__main__':
#     video_path = 'your_video.mp4'
#     reader = ReadArray(video_path)
#     print(f"Total frames: {len(reader)}")
#     frame = reader[10]  # 取得第 10 個影格
#     print(frame.shape)


class ReadArray:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.container = av.open(video_path)
        self.video_stream = self.container.streams.video[0]
        self.fps = self.video_stream.average_rate
        tb = self.video_stream.time_base
        ar = self.video_stream.average_rate
        self.pts2index = tb * ar
        self.total_frames = 0
        self.index = 0
        self.__len__()

    def __len__(self):
        if self.total_frames != 0:
            return self.total_frames
        
        n = self.video_stream.frames
        for i in range(n - 100, n):
            try:
                _ = self.__getitem__(i)
            except:
                return i
        self.total_frames = self.video_stream.frames
        return self.total_frames

    
    def __getitem__(self, frame_index):
        if frame_index == self.index + 1:
            try:
                self.index = frame_index
                frame = next(self.container.decode(self.video_stream))
                return frame.to_rgb().to_ndarray()
            except:
                pass
        

        timestamp = frame_index / self.pts2index
        self.container.seek(int(timestamp), stream=self.video_stream, backward=True)
        for frame in self.container.decode(self.video_stream):
            index = int(frame.pts * self.pts2index)
            if index == frame_index:
                self.index = frame_index
                break
        return frame.to_rgb().to_ndarray()

    def __del__(self):
        self.container.close()



        
        
        
class WriteArray:
    def __init__(self, video_path, fps, crf=15):
        self.output = av.open(video_path, 'w')
        self.stream = self.output.add_stream('libx264', rate=fps)
        self.stream.options = {'crf': str(crf)}
        self.stream.pix_fmt = 'yuv420p'
        self.init = False
        self._closed = False

    def append(self, frame): #frame: ndarray, H, W, C
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected frame with shape (H, W, 3)")

        if not self.init:
            self.stream.height, self.stream.width = frame.shape[:2]
            self.init = True
            
        frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        
        for packet in self.stream.encode(frame):
            self.output.mux(packet)
        
    def close(self):
        if self._closed:
            return
        for packet in self.stream.encode():
            self.output.mux(packet)
        self.output.close()
        self._closed = True

    def __del__(self):
        self.close()