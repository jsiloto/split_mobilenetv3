import gdown
import wget
import zipfile
from pathlib import Path


# Path("./checkpoints").mkdir(parents=True, exist_ok=True)

gdown.download('https://drive.google.com/uc?id=1wnBBQYG21b_rvGlmDi9kboTxcDlbRV2K',
               'checkpoints.zip')

with zipfile.ZipFile('checkpoints.zip', 'r') as zip_ref:
    zip_ref.extractall('.')