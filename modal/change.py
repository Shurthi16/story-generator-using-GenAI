import os
from PIL import Image

# Input and output folders
input_folder = r"C:\Users\Murugan\cartoon_story_generator\data\raw_images\story image test"
output_folder = r"C:\Users\Murugan\cartoon_story_generator\data\converted_images"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

# Loop through all files in input folder
for filename in os.listdir(input_folder):
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext in valid_extensions:
        input_path = os.path.join(input_folder, filename)
        
        # Open and convert to RGB (to avoid transparency issues)
        with Image.open(input_path) as img:
            rgb_image = img.convert("RGB")
            output_filename = os.path.splitext(filename)[0] + ".jpg"
            output_path = os.path.join(output_folder, output_filename)
            
            rgb_image.save(output_path, "JPEG")

print("✅ Conversion complete. All images saved as JPG in:", output_folder)
