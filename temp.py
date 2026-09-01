from dotenv import load_dotenv
import os

# Load variables from the .env file
load_dotenv()

# Get the Hugging Face token
hf_token = os.getenv("HF_TOKEN")

# Check whether it was loaded
if hf_token:
    print("Token loaded successfully!")
    print("Token starts with:", hf_token[:6] + "...")
else:
    print("ERROR: HF_TOKEN was not found.")
    print("Make sure your .env file contains:")
    print("HF_TOKEN=hf_your_token_here")