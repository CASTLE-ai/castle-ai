import os
import subprocess


def download_file(url, destination):
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(os.path.dirname(destination))
    # Check if the file does not exist before downloading
    if not os.path.isfile(destination):
        print(f"Downloading {os.path.basename(destination)}...")
        subprocess.run(["wget", "-P", os.path.dirname(destination), url])
    else:
        print(f"{os.path.basename(destination)} already downloaded.")

def download_with_gdown(file_id, destination):
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if the file does not exist before downloading
    if not os.path.isfile(destination):
        print(f"Downloading {os.path.basename(destination)}...")
        subprocess.run(['gdown', file_id, '--output', destination])
    else:
        print(f"{os.path.basename(destination)} already downloaded.")
