import urllib.request
import tarfile
import os
import ssl

# S3 Mirror used by Keras
DATA_URL = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"
DATA_DIR = "data"
TAR_PATH = os.path.join(DATA_DIR, "babi.tar.gz")

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Downloading bAbI dataset from {DATA_URL}...")
    # Bypass SSL verification if needed (sometimes helps with older python/systems)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    with urllib.request.urlopen(DATA_URL, context=ctx) as u, open(TAR_PATH, 'wb') as f:
        f.write(u.read())
        
    print("Download complete.")
    
    print("Extracting...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    print("Extraction complete.")

    # List extracted files to verify
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if "qa15" in file:
                print(f"Found Task 15 file: {os.path.join(root, file)}")

if __name__ == "__main__":
    download_and_extract()
