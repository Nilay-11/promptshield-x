import time
import requests
from pathlib import Path

DEST = Path(r"C:\Users\nilay\.cache\huggingface\hub\models--microsoft--deberta-v3-base\snapshots\8ccc9b6f36199bec6961081d44eb72fb3f7353f3\pytorch_model.bin")
URL = "https://huggingface.co/microsoft/deberta-v3-base/resolve/main/pytorch_model.bin"

def download_weights():
    print(f"Downloading {URL} to {DEST}...")
    t0 = time.time()
    resp = requests.get(URL, stream=True, timeout=60)
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(DEST, "wb") as f:
        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                pct = (downloaded / total) * 100 if total else 0
                mb = downloaded / (1024 * 1024)
                tot_mb = total / (1024 * 1024)
                print(f"\rDownloaded {mb:.1f}/{tot_mb:.1f} MB ({pct:.1f}%) in {time.time()-t0:.1f}s", end="", flush=True)
    print("\nDownload finished successfully!")

if __name__ == "__main__":
    download_weights()
