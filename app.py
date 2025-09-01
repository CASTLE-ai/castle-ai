import os
from argparse import ArgumentParser
import platform




from castle.ui import create_ui

OS_SYS = platform.uname().system
COLAB_GPU = 'COLAB_GPU' in os.environ
SHARE = True

parser = ArgumentParser()
parser.add_argument("--project-folder", dest="root")
# parser.add_argument("--video_storage_path", dest="root") # TODO
args = parser.parse_args()

app = create_ui(OS_SYS, args.root)

if __name__ == '__main__':
     app.queue(max_size=20)
     
     # 設定 allowed_paths 來解決 Colab 中的路徑權限問題
     allowed_paths = []
     if COLAB_GPU:  # 在 Colab 環境中
          allowed_paths = [
               "/content/drive/MyDrive/castle-projects",  # Google Drive 專案目錄
               "/tmp",  # 臨時目錄
               "/content"  # Colab 內容目錄
          ]
     
     app.launch(
          server_name='0.0.0.0', 
          share=COLAB_GPU or SHARE,
          allowed_paths=allowed_paths if allowed_paths else None
     )