import os
import zipfile
import requests
from urllib.parse import urlparse
import shutil

url = "https://zenodo.org/records/2867216/files/26_29_09_2017_KCL.zip?download=1"

download_dir = "downloaded_audio"
extract_dir = "extracted_audio"
output_dir = "all_audio_files"

os.makedirs(download_dir, exist_ok=True)
os.makedirs(extract_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def download_file(url, download_dir):
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path.split('?')[0])
    filepath = os.path.join(download_dir, filename)
    
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to {filepath}")
    return filepath

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")

def collect_wav_files(source_dir, output_dir):
    print("Collecting all .wav files...")
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                src_path = os.path.join(root, file)
                
                dest_path = os.path.join(output_dir, file)
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(file)
                    dest_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
                    counter += 1
                
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {src_path} -> {dest_path}")

try:
    zip_path = download_file(url, download_dir)
    extract_zip(zip_path, extract_dir)
    collect_wav_files(extract_dir, output_dir)
    
    print(f"\nAll audio files have been collected in: {os.path.abspath(output_dir)}")
    print(f"Total files: {len(os.listdir(output_dir))}")
except Exception as e:
    print(f"An error occurred: {e}")