import gdown
import torch
from pathlib import Path

def download_model(gdrive_url: str, save_path: str = "connect_4_bot.pt"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {gdrive_url} ...")
    gdown.download(gdrive_url, str(save_path), quiet=False)

    checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
    if "policy" in checkpoint:
        print("Saving state_dict only...")
        torch.save(checkpoint["policy"].state_dict(), save_path)

    print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="Google Drive URL of the checkpoint")
    parser.add_argument("--output", type=str, default="connect_4_bot.pt", help="Where to save locally")
    args = parser.parse_args()

    download_model(args.url, args.output)