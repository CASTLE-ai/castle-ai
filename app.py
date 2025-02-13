import os
from argparse import ArgumentParser
import platform




from castle.ui import create_ui

OS_SYS = platform.uname().system
COLAB_GPU = 'COLAB_GPU' in os.environ
SHARE = False

parser = ArgumentParser()
parser.add_argument("--project-folder", dest="root")
# parser.add_argument("--video_storage_path", dest="root") # TODO
args = parser.parse_args()

app = create_ui(OS_SYS, args.root)

if __name__ == '__main__':
     app.queue(max_size=20)
     app.launch(server_name='0.0.0.0', share=COLAB_GPU or SHARE)