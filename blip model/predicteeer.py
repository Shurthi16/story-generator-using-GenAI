import torch
from PIL import Image
from transformers import BlipProcessor, GPT2Tokenizer
from traineeer import ImageToStoryModel  # Make sure traineeer.py is in the same directory
import os

# === CONFIG ===
MODEL_PATH = "blip model/training_output/models/best_model.pth"  # Path to best_model.pth
IMAGE_PATH = r"C:\Users\Murugan\cartoon_story_generator\data\converted_images\265.jpg"  # Path to your sample test image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load Pre-trained Tokenizer & Processor ===
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_tokenizer.add_special_tokens({"bos_token": "<bos>"})

# === Load Model ===
model = ImageToStoryModel()
model.gpt.resize_token_embeddings(len(gpt_tokenizer))  # Resize for <bos> token
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# === Load and Preprocess Image ===
image = Image.open(IMAGE_PATH).convert("RGB")
blip_inputs = blip_processor(image, return_tensors="pt").to(DEVICE)
pixel_values = blip_inputs['pixel_values']

# === Generate Story ===
with torch.no_grad():
    story = model.generate_story(pixel_values, tokenizer=gpt_tokenizer, max_length=150)

# === Output ===
print("\n📘 Generated Story:")
print(story)
