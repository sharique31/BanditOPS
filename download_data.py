from pathlib import Path
import os

from dotenv import load_dotenv
from turing_deinterleaving_challenge import download_dataset


# Load Hugging Face token from .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

hf_token = os.getenv("HUGGING_FACE_TOKEN")

if not hf_token:
    raise ValueError("HUGGING_FACE_TOKEN was not found in the .env file")


# Download only the test subset first
download_dataset(
    save_dir=Path("./data"),
    subsets=["test"],
    hf_token=hf_token
)

print("Download completed!")