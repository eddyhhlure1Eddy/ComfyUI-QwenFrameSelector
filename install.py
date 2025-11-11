"""
Installation script for ComfyUI-QwenFrameSelector
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print("[QwenFrameSelector] requirements.txt not found, skipping dependency installation")
        return
    
    try:
        print("[QwenFrameSelector] Installing dependencies...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_path
        ])
        print("[QwenFrameSelector] Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[QwenFrameSelector] Error installing dependencies: {e}")
        print("[QwenFrameSelector] You may need to install them manually:")
        print("  pip install requests pillow numpy torch")

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("[QwenFrameSelector] FFmpeg is installed")
            return True
        else:
            print("[QwenFrameSelector] FFmpeg check failed")
            return False
    except FileNotFoundError:
        print("[QwenFrameSelector] WARNING: FFmpeg not found in PATH!")
        print("[QwenFrameSelector] Please install FFmpeg:")
        print("  Windows: winget install FFmpeg")
        print("  Linux: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        return False

if __name__ == "__main__":
    print("[QwenFrameSelector] Starting installation...")
    install_requirements()
    check_ffmpeg()
    print("[QwenFrameSelector] Installation complete!")
    print("[QwenFrameSelector] Don't forget to get your OpenRouter API key at https://openrouter.ai/")
