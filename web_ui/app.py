import os
import platform


from castle.ui import create_ui

OS_SYS = platform.uname().system
COLAB_GPU = 'COLAB_GPU' in os.environ
SHARE = False


app = create_ui(OS_SYS)

if __name__ == '__main__':
     app.queue(concurrency_count=10)
     app.launch(server_name='0.0.0.0', share=COLAB_GPU or SHARE).queue()