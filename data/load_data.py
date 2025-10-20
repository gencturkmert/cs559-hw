import requests
import os
import zipfile

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"File downloaded successfully: {output_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

if __name__ == "__main__":
    url = "http://www.cs.bilkent.edu.tr/~dibeklioglu/teaching/cs559/docs/SCUT_FBP5500_downsampled.zip"
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "SCUT_FBP5500_downsampled.zip")
    
    download_file(url, output_path)
    
    # Unzip the downloaded file
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"File unzipped successfully to: {output_dir}")