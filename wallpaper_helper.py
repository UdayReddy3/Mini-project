"""
Wallpaper Helper - Automatically detects and loads wallpaper images
Place your wallpaper in: assets/wallpapers/
"""

import os
from pathlib import Path
import base64

def get_wallpaper_path():
    """Find the first wallpaper image in assets/wallpapers/"""
    wallpaper_dir = Path("assets/wallpapers")
    
    if not wallpaper_dir.exists():
        return None
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    
    # Find first image file
    for file in sorted(wallpaper_dir.glob('*')):
        if file.suffix.lower() in supported_formats:
            return str(file)
    
    return None

def encode_image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def get_wallpaper_css(image_path):
    """Generate CSS for wallpaper background"""
    if not image_path or not os.path.exists(image_path):
        return None
    
    try:
        # Encode image to base64
        base64_img = encode_image_to_base64(image_path)
        
        if not base64_img:
            return None
        
        # Determine file type from path
        file_ext = os.path.splitext(image_path)[1].lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.gif': 'image/gif'
        }.get(file_ext, 'image/jpeg')
        
        # Return CSS with embedded image
        css = f"""
        body, .stApp {{
            background-image: url('data:{mime_type};base64,{base64_img}');
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
            min-height: 100vh;
        }}
        """
        return css
    
    except Exception as e:
        print(f"Error generating CSS: {e}")
        return None

# Auto-detect wallpaper
WALLPAPER_PATH = get_wallpaper_path()

if __name__ == "__main__":
    print("=== Wallpaper Detector ===")
    print(f"Wallpaper found: {WALLPAPER_PATH}")
    if WALLPAPER_PATH:
        print(f"Path: {os.path.abspath(WALLPAPER_PATH)}")
        print(f"Size: {os.path.getsize(WALLPAPER_PATH) / 1024 / 1024:.2f} MB")
