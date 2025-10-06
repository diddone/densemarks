#!/usr/bin/env python3
"""
Simple script to upload DenseMarks model to Hugging Face Hub.
Uploads only: README.md and model.safetensors (model weights only).
"""

import os
import torch
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from safetensors.torch import save_file

# Configuration
HF_USERNAME = "diddone"
HF_REPO_NAME = "densemarks"
FULL_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

def create_model_safetensors():
    """
    Extract only the 'model' key from model_final.pth and save as model.safetensors.
    This makes it compatible with HuggingFace's standard model loading using SafeTensors format.
    """
    print("Creating model.safetensors from model_final.pth...")

    # Load the original checkpoint
    checkpoint = torch.load("model_final.pth", map_location='cpu')

    print(f"  Original checkpoint keys: {list(checkpoint.keys())}")

    # Extract only the model weights
    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
        print(f"  Extracted model state dict with {len(model_state_dict)} parameters")

        # Save as model.safetensors
        save_file(model_state_dict, "model.safetensors")
        print("  ✓ Created model.safetensors")
        return True
    else:
        print("  ✗ Error: 'model' key not found in checkpoint")
        return False

def upload_to_hf():
    """Upload files to Hugging Face Hub."""
    print(f"\nUploading to {FULL_REPO_ID}...")

    # Check for HF token
    token = os.getenv("HF_TOKEN")
    if not token:
        print("\n⚠ Error: HF_TOKEN environment variable not set")
        print("Please run: export HF_TOKEN=your_token_here")
        print("Or login with: huggingface-cli login")
        return False

    try:
        api = HfApi()

        # Create repository (or get existing)
        print("  Creating/accessing repository...")
        try:
            create_repo(
                repo_id=FULL_REPO_ID,
                token=token,
                private=False,
                repo_type="model",
                license="cc-by-nc-4.0",
                exist_ok=True
            )
            print("  ✓ Repository ready")
        except Exception as e:
            print(f"  Note: {e}")

        # Upload individual files
        files_to_upload = [
            "README.md",
            "model.safetensors"
        ]

        print("\n  Uploading files...")
        for file in files_to_upload:
            if os.path.exists(file):
                api.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=file,
                    repo_id=FULL_REPO_ID,
                    token=token,
                )
                print(f"    ✓ Uploaded {file}")
            else:
                print(f"    ✗ Warning: {file} not found")

        print(f"\n🎉 Upload successful!")
        print(f"Repository URL: https://huggingface.co/{FULL_REPO_ID}")
        return True

    except Exception as e:
        print(f"\n✗ Error during upload: {e}")
        return False

def cleanup():
    """Remove temporary files."""
    print("\nCleaning up temporary files...")
    if os.path.exists("model.safetensors"):
        os.remove("model.safetensors")
        print("  ✓ Removed model.safetensors")

def main():
    print("=" * 60)
    print("DenseMarks Model Upload to Hugging Face")
    print("=" * 60)

    # Step 1: Create model.safetensors (only model weights)
    if not create_model_safetensors():
        print("\n✗ Failed to create model.safetensors")
        return

    # Step 2: Upload to HuggingFace
    success = upload_to_hf()

    # Step 3: Cleanup temporary files
    cleanup()

    if success:
        print("\n" + "=" * 60)
        print("✓ All done! Your model is now on Hugging Face.")
        print("=" * 60)

if __name__ == "__main__":
    main()