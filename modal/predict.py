import os
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel, GPT2Tokenizer, GPT2LMHeadModel

# CHANGE THIS PATH to your test image
TEST_IMAGE_PATH = r"C:\Users\Murugan\cartoon_story_generator\data\converted_images\214.jpg"

def simple_test():
    print("=== Simple Image Story Test ===")
    
    # Load models
    print("Loading models...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    text_decoder = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Try to load trained weights if available
    model_path = r"C:\Users\Murugan\cartoon_story_generator\outvit\best_model.pth"
    if os.path.exists(model_path):
        try:
            text_decoder.load_state_dict(torch.load(model_path, map_location='cpu'))
            print("✓ Loaded trained model weights")
        except:
            print("⚠ Could not load trained weights, using base model")
    else:
        print("⚠ No trained weights found, using base GPT2")
    
    # Check if image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Image not found: {TEST_IMAGE_PATH}")
        print("Please update TEST_IMAGE_PATH to point to your test image")
        return
    
    # Load and process image
    print(f"Loading image: {TEST_IMAGE_PATH}")
    try:
        image = Image.open(TEST_IMAGE_PATH).convert("RGB")
        print(f"✓ Image loaded successfully: {image.size}")
        
        # Process image
        image_inputs = image_processor(images=image, return_tensors="pt")
        image_tensor = image_inputs["pixel_values"]
        
        # Get image features
        # Get image features
        with torch.no_grad():
            image_features = image_encoder(pixel_values=image_tensor)
            img_embedding = image_features.last_hidden_state[:, 0, :]  # Shape: [1, 768]

        # Display the image features before passing to GPT-2
        try:
            # Expand [CLS] embedding into a pseudo-sequence for GPT2
            direct_features = img_embedding.unsqueeze(1).repeat(1, 20, 1)  # [1, 20, 768]

            with torch.no_grad():
                outputs_direct = text_decoder.generate(
                    inputs_embeds=direct_features,
                    max_length=30,
                    num_return_sequences=1,
                    temperature=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            vit_text_direct = tokenizer.decode(outputs_direct[0], skip_special_tokens=True)
            print(f"📝 ViT → Text: {vit_text_direct}")

        except Exception as e:
            print(f"❌ Failed to decode directly from ViT: {e}")

        
        print(f"✓ Image processed, feature shape: {img_embedding.shape}")
        
        # Generate text
        print("Generating story...")
        
        # Method 1: Simple generation with prompt
        prompt = "This image shows"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = text_decoder.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n=== Results ===")
        print(f"Image: {os.path.basename(TEST_IMAGE_PATH)}")
        print(f"Generated text: {generated_text}")
        
        # Method 2: Try to use image features (experimental)
        print("\n=== Experimental: Using image features ===")
        try:
            # Expand image features to create a sequence
            seq_length = 20
            expanded_features = img_embedding.unsqueeze(1).repeat(1, seq_length, 1)
            
            with torch.no_grad():
                outputs = text_decoder.generate(
                    inputs_embeds=expanded_features,
                    max_length=seq_length + 10,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            experimental_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Experimental generation: {experimental_text}")
            
        except Exception as e:
            print(f"Experimental method failed: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    simple_test()