"""
Complete Image-to-Story Training Pipeline
Self-contained training script for image captioning/story generation
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import csv
from datetime import datetime

from transformers import (
    ViTImageProcessor, 
    ViTModel, 
    GPT2Tokenizer, 
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModel
)

class ImageStoryDataset(Dataset):
    """Dataset for loading images and their corresponding text descriptions."""
    
    def __init__(self, csv_file, image_dir, image_processor, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file, delimiter='|', skipinitialspace=True)
        self.image_dir = Path(image_dir)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Auto-detect column names
        self.image_col = self._find_column(['image', 'filename', 'img', 'image_name', 'file'])
        self.text_col = self._find_column(['caption', 'story', 'text', 'description', 'comment'])
        
        # Clean data
        self.data = self.data.dropna(subset=[self.image_col, self.text_col])
        self.data = self.data[self.data[self.text_col].str.strip() != '']
        
        print(f"Dataset loaded: {len(self.data)} samples")
        print(f"Image column: {self.image_col}, Text column: {self.text_col}")
    
    def _find_column(self, possible_names):
        for col in self.data.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                return col
        return self.data.columns[0] if len(self.data.columns) > 0 else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image_path = self.image_dir / row[self.image_col]
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        except:
            # Fallback to blank image
            image = Image.new('RGB', (224, 224), 'white')
            pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        
        # Process text
        text = str(row[self.text_col]).strip()
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': pixel_values,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text
        }

class ImageToStoryModel(nn.Module):
    """Complete Image-to-Story generation model."""
    
    def __init__(self, vision_dim=768, text_dim=768, hidden_dim=512):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vision_projection = nn.Linear(vision_dim, hidden_dim)
        
        # Text decoder
        self.text_decoder = GPT2LMHeadModel.from_pretrained('gpt2')
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Fusion layers
        self.multimodal_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.output_projection = nn.Linear(hidden_dim, self.text_decoder.config.vocab_size)
        
    def forward(self, pixel_values, input_ids, attention_mask=None):
        batch_size = pixel_values.size(0)
        
        # Encode images
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state  # [batch, seq_len, dim]
        image_features = self.vision_projection(image_features)
        
        # Get text embeddings
        text_embeds = self.text_decoder.transformer.wte(input_ids)
        text_embeds = self.text_projection(text_embeds)
        
        # Multimodal fusion
        fused_features, _ = self.multimodal_fusion(
            query=text_embeds,
            key=image_features,
            value=image_features
        )
        
        # Combine and normalize
        combined = self.layer_norm(text_embeds + self.dropout(fused_features))
        
        # Generate output logits
        logits = self.output_projection(combined)
        
        return logits

class TrainingMetrics:
    """Track and save training metrics."""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
        
        # Initialize CSV
        self.csv_file = self.save_dir / 'metrics.csv'
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr', 'timestamp'])
    
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
        
        # Save to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc, lr, datetime.now()])
    
    def plot_and_save(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0,0].plot(self.history['epoch'], self.history['train_loss'], 'b-', label='Train')
        axes[0,0].plot(self.history['epoch'], self.history['val_loss'], 'r-', label='Validation')
        axes[0,0].set_title('Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Accuracy plot
        axes[0,1].plot(self.history['epoch'], self.history['train_acc'], 'g-', label='Train')
        axes[0,1].plot(self.history['epoch'], self.history['val_acc'], 'orange', label='Validation')
        axes[0,1].set_title('Accuracy')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Learning rate
        axes[1,0].plot(self.history['epoch'], self.history['lr'], 'purple')
        axes[1,0].set_title('Learning Rate')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('LR')
        axes[1,0].grid(True)
        
        # Loss difference
        loss_diff = [abs(t-v) for t, v in zip(self.history['train_loss'], self.history['val_loss'])]
        axes[1,1].plot(self.history['epoch'], loss_diff, 'brown')
        axes[1,1].set_title('Train-Val Loss Difference')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('|Train Loss - Val Loss|')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

def calculate_accuracy(logits, labels, ignore_index=-100):
    """Calculate token-level accuracy."""
    predictions = logits.argmax(dim=-1)
    mask = (labels != ignore_index)
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item() if accuracy.numel() > 0 else 0.0

def save_model(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }, filepath)

def train_epoch(model, dataloader, optimizer, criterion, device, tokenizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(pixel_values, input_ids, attention_mask)
        
        # Calculate loss (shift for next-token prediction)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Calculate accuracy
        acc = calculate_accuracy(shift_logits, shift_labels, tokenizer.pad_token_id)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
        
        # Memory cleanup
        del pixel_values, input_ids, attention_mask, logits, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return total_loss / num_batches, total_acc / num_batches

def validate_epoch(model, dataloader, criterion, device, tokenizer):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = model(pixel_values, input_ids, attention_mask)
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Calculate accuracy
            acc = calculate_accuracy(shift_logits, shift_labels, tokenizer.pad_token_id)
            
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
            
            # Memory cleanup
            del pixel_values, input_ids, attention_mask, logits, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return total_loss / num_batches, total_acc / num_batches

def main():
    parser = argparse.ArgumentParser(description='Train Image-to-Story Model')
    
    # Data paths - REPLACE WITH YOUR ACTUAL PATHS
    parser.add_argument('--image_dir', type=str, default=r"C:\Users\Murugan\cartoon_story_generator\newflickrdataset\flickr30k_images", 
                       help='Directory containing images')
    parser.add_argument('--csv_file', type=str, default=r"C:\Users\Murugan\cartoon_story_generator\newflickrdataset\results.csv",
                       help='CSV file with image names and captions')
    parser.add_argument('--output_dir', type=str, default='./outmodal',
                       help='Output directory')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=5)
    
    # Resume training
    parser.add_argument('--resume', type=str, help='Path to checkpoint')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / 'flickerout'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    # Create dataset
    print("Loading dataset...")
    full_dataset = ImageStoryDataset(args.csv_file, args.image_dir, image_processor, tokenizer, args.max_length)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    model = ImageToStoryModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Metrics tracker
    metrics = TrainingMetrics(output_dir)
    
    # Resume if needed
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('metrics', {}).get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, tokenizer)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, tokenizer)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics
        metrics.update(epoch+1, train_loss, val_loss, train_acc, val_acc, current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, optimizer, epoch, {'best_val_acc': best_val_acc}, output_dir / 'best_model.pth')
            print(f"🎉 New best model saved! Accuracy: {best_val_acc:.4f}")
        
        # Regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_model(model, optimizer, epoch, {'val_acc': val_acc}, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Plot metrics
        if (epoch + 1) % 10 == 0:
            metrics.plot_and_save()
    
    # Final save
    save_model(model, optimizer, args.epochs-1, {'final_val_acc': val_acc}, output_dir / 'final_model.pth')
    metrics.plot_and_save()
    
    print(f"\n✅ Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved in: {output_dir}")

if __name__ == '__main__':
    main()