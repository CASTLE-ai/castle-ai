import os
from argparse import ArgumentParser
import platform




from castle.ui import create_ui

OS_SYS = platform.uname().system
COLAB_GPU = 'COLAB_GPU' in os.environ
SHARE = False

parser = ArgumentParser()
parser.add_argument("--project-folder", dest="root")
args = parser.parse_args()

app = create_ui(OS_SYS, args.root)

if __name__ == '__main__':
     app.queue()
     app.launch(server_name='0.0.0.0', share=COLAB_GPU or SHARE).queue()