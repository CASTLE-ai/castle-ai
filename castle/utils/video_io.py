import av
import os

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
    def __init__(self, video_path, fps, crf=18):
        self.output = av.open(video_path, 'w')
        self.stream = self.output.add_stream('libx264', rate=fps)
        self.stream.options = {'crf': str(crf)}
        self.stream.pix_fmt = 'yuv420p'
        self.init = False

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
        for packet in self.stream.encode():
            self.output.mux(packet)
            
        self.output.close()