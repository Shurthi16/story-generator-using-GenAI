import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    GPT2LMHeadModel, GPT2Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import warnings
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from collections import defaultdict
import math

warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccuracyMetrics:
    """Class to calculate various accuracy metrics for text generation"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
    def calculate_bleu_score(self, reference, hypothesis):
        """Calculate BLEU score between reference and hypothesis"""
        try:
            # Tokenize sentences
            reference_tokens = word_tokenize(reference.lower())
            hypothesis_tokens = word_tokenize(hypothesis.lower())
            
            # Calculate BLEU score (sentence-level)
            bleu_score = sentence_bleu(
                [reference_tokens], 
                hypothesis_tokens, 
                smoothing_function=self.smoothing_function
            )
            return bleu_score * 100  # Convert to percentage
        except:
            return 0.0
    
    def calculate_rouge_scores(self, reference, hypothesis):
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {
                'rouge1': scores['rouge1'].fmeasure * 100,
                'rouge2': scores['rouge2'].fmeasure * 100,
                'rougeL': scores['rougeL'].fmeasure * 100
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_token_accuracy(self, predicted_tokens, target_tokens, pad_token_id):
        """Calculate token-level accuracy"""
        # Remove padding tokens
        valid_mask = (target_tokens != pad_token_id) & (target_tokens != -100)
        if valid_mask.sum() == 0:
            return 0.0
        
        correct_predictions = (predicted_tokens == target_tokens) & valid_mask
        accuracy = correct_predictions.sum().item() / valid_mask.sum().item()
        return accuracy * 100  # Convert to percentage
    
    def calculate_perplexity(self, loss):
        """Calculate perplexity from loss"""
        try:
            return math.exp(loss)
        except:
            return float('inf')

class ImageStoryDataset(Dataset):
    def __init__(self, csv_path, image_dir, blip_processor, gpt_tokenizer, max_story_length=512):
        """
        Dataset for image-to-story generation
        csv_path: path to CSV file with columns ['image_filename', 'story']
        image_dir: directory containing images
        """
        print(f"Attempting to load CSV: {csv_path}")
        
        # First, let's examine the file
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_few_lines = [f.readline().strip() for _ in range(5)]
            print("First 5 lines of CSV:")
            for i, line in enumerate(first_few_lines, 1):
                print(f"Line {i}: {repr(line)}")
        except Exception as e:
            print(f"Could not read file with utf-8: {e}")
        
        # Try different encodings and parsing options
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        self.df = None
        
        for encoding in encodings:
            try:
                print(f"Trying encoding: {encoding}")
                # Try with different parsing options
                parsing_options = [
                    {'encoding': encoding},
                    {'encoding': encoding, 'sep': ',', 'quotechar': '"'},
                    {'encoding': encoding, 'sep': ',', 'quotechar': '"', 'escapechar': '\\'},
                    {'encoding': encoding, 'sep': ',', 'on_bad_lines': 'skip'},
                    {'encoding': encoding, 'sep': ';'},  # Sometimes semicolon is used
                ]
                
                for i, options in enumerate(parsing_options):
                    try:
                        print(f"  Trying parsing option {i+1}: {options}")
                        self.df = pd.read_csv(csv_path, **options)
                        print(f"Successfully loaded CSV with {encoding} encoding and option {i+1}")
                        break
                    except Exception as parse_error:
                        print(f"    Failed: {parse_error}")
                        continue
                
                if self.df is not None:
                    break
                    
            except Exception as e:
                print(f"Failed with {encoding}: {e}")
                continue
        
        if self.df is None:
            print(f"\nERROR: Could not read CSV file {csv_path}")
            print("Please check your CSV file format. It should look like:")
            print("image_filename,story")
            print("image1.jpg,\"Once upon a time there was a story...\"")
            print("image2.jpg,\"Another story with, commas in it...\"")
            raise ValueError(f"Could not read CSV file {csv_path} with any of the tried encodings and options")
        
        print(f"Dataset loaded: {len(self.df)} samples")
        print(f"CSV columns: {list(self.df.columns)}")
        print(f"CSV shape: {self.df.shape}")
        
        # Show first few rows
        print("First few rows:")
        print(self.df.head())
        
        # Rename columns if necessary
        column_renames = {
            'image path ': 'image_filename',
            'text description ': 'story'
        }
        self.df.rename(columns=column_renames, inplace=True)

        # Check if required columns exist after renaming
        required_columns = ['image_filename', 'story']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"Available columns after renaming: {list(self.df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}.")

        
        # Clean up data
        self.df = self.df.dropna()  # Remove rows with missing values
        print(f"After removing NaN values: {len(self.df)} samples")
        self.image_dir = image_dir
        self.blip_processor = blip_processor
        self.gpt_tokenizer = gpt_tokenizer
        self.max_story_length = max_story_length
        
        # Add special tokens if not present
        if self.gpt_tokenizer.pad_token is None:
            self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        special_tokens = {"bos_token": "<bos>"}
        self.gpt_tokenizer.add_special_tokens(special_tokens)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_filename'])
        story = row['story']
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
            blip_inputs = self.blip_processor(image, return_tensors="pt")
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Create dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
            blip_inputs = self.blip_processor(image, return_tensors="pt")
        
        # Tokenize story for GPT
        story_text = f"{self.gpt_tokenizer.bos_token}{story}{self.gpt_tokenizer.eos_token}"
        story_tokens = self.gpt_tokenizer.encode(
            story_text, 
            max_length=self.max_story_length, 
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': blip_inputs['pixel_values'].squeeze(0),
            'story_tokens': story_tokens.squeeze(0),
            'story_text': story
        }

class ImageToStoryModel(nn.Module):
    def __init__(self, blip_model_name="Salesforce/blip-image-captioning-base", 
                 gpt_model_name="gpt2", hidden_dim=768, fusion_dim=512):
        super().__init__()
        
        # Load pre-trained models
        self.blip = BlipForConditionalGeneration.from_pretrained(blip_model_name)
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        
        # Freeze BLIP parameters initially (optional - can be unfrozen later)
        for param in self.blip.parameters():
            param.requires_grad = False
            
        # Get dimensions
        self.blip_hidden_size = self.blip.config.text_config.hidden_size
        self.gpt_hidden_size = self.gpt.config.hidden_size
        
        # Fusion layers to combine image and text features
        self.image_projection = nn.Linear(self.blip_hidden_size, fusion_dim)
        self.text_projection = nn.Linear(self.gpt_hidden_size, fusion_dim)
        self.fusion_layer = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        self.output_projection = nn.Linear(fusion_dim, self.gpt_hidden_size)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, pixel_values, input_ids, attention_mask=None):
        batch_size = pixel_values.size(0)
        
        # Extract image features using BLIP
        with torch.no_grad():
            blip_outputs = self.blip.vision_model(pixel_values)
            image_features = blip_outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_size]
        
        # Get GPT embeddings
        gpt_embeddings = self.gpt.transformer.wte(input_ids)  # [batch, seq_len, hidden_size]
        
        # Project features to fusion dimension
        image_proj = self.image_projection(image_features).unsqueeze(1)  # [batch, 1, fusion_dim]
        text_proj = self.text_projection(gpt_embeddings)  # [batch, seq_len, fusion_dim]
        
        # Concatenate image and text features
        combined_features = torch.cat([image_proj, text_proj], dim=1)  # [batch, seq_len+1, fusion_dim]
        combined_features = self.layer_norm(combined_features)
        
        # Apply fusion attention
        fused_features, _ = self.fusion_layer(combined_features, combined_features, combined_features)
        fused_features = self.dropout(fused_features)
        
        # Project back to GPT dimension and remove image token
        text_features = self.output_projection(fused_features[:, 1:, :])  # Remove image token
        
        # Pass through GPT with modified embeddings
        outputs = self.gpt(inputs_embeds=text_features, attention_mask=attention_mask)
        
        return outputs
    
    def generate_story(self, pixel_values, tokenizer, max_length=200, temperature=0.8, top_p=0.9):
        """Generate story from image"""
        self.eval()
        with torch.no_grad():
            # Start with BOS token
            input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(pixel_values.device)
            generated_tokens = []
            
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(pixel_values, input_ids)
                logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-p sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generated_text

class ImageStoryTrainer:
    def __init__(self, model, train_loader, val_loader, gpt_tokenizer, device, save_dir="./training_output"):
        self.gpt_tokenizer = gpt_tokenizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        # Initialize accuracy metrics calculator
        self.accuracy_metrics = AccuracyMetrics()
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []  # Token accuracy
        self.val_accuracies = []   # Token accuracy
        self.val_bleu_scores = []
        self.val_rouge_scores = []
        self.val_perplexities = []
        self.best_val_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_model_path = None
        
        # Setup logging
        logging.basicConfig(
            filename=os.path.join(save_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_optimizer(self, learning_rate=5e-5, weight_decay=0.01, warmup_steps=1000, total_steps=10000):
        """Setup optimizer and scheduler"""
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} - Training")
        
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(self.device)
            story_tokens = batch['story_tokens'].to(self.device)
            
            # Create attention mask
            attention_mask = (story_tokens != self.gpt_tokenizer.pad_token_id).long()
            
            # Create labels (shift right for causal LM)
            labels = story_tokens.clone()
            labels[labels == self.gpt_tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=story_tokens,
                attention_mask=attention_mask
            )
            
            # Calculate loss
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Calculate token accuracy
            predicted_tokens = torch.argmax(shift_logits, dim=-1)
            batch_accuracy = self.accuracy_metrics.calculate_token_accuracy(
                predicted_tokens, shift_labels, self.gpt_tokenizer.pad_token_id
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_accuracy += batch_accuracy
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{batch_accuracy:.2f}%"
            })
            
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)
        return avg_loss, avg_accuracy
    
    def validate(self, epoch):
        """Validate the model with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_bleu = 0
        total_rouge1 = 0
        total_rouge2 = 0
        total_rougeL = 0
        num_batches = 0
        num_generated_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} - Validation")
            
            for i, batch in enumerate(progress_bar):
                pixel_values = batch['pixel_values'].to(self.device)
                story_tokens = batch['story_tokens'].to(self.device)
                story_texts = batch['story_texts']
                
                attention_mask = (story_tokens != self.gpt_tokenizer.pad_token_id).long()
                labels = story_tokens.clone()
                labels[labels == self.gpt_tokenizer.pad_token_id] = -100
                
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=story_tokens,
                    attention_mask=attention_mask
                )
                
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Calculate token accuracy
                predicted_tokens = torch.argmax(shift_logits, dim=-1)
                batch_accuracy = self.accuracy_metrics.calculate_token_accuracy(
                    predicted_tokens, shift_labels, self.gpt_tokenizer.pad_token_id
                )
                
                total_loss += loss.item()
                total_accuracy += batch_accuracy
                num_batches += 1
                
                # Generate stories for BLEU/ROUGE evaluation (sample a few batches)
                if i < 5:  # Only evaluate first 5 batches to save time
                    for j in range(min(2, len(pixel_values))):  # Max 2 samples per batch
                        try:
                            generated_story = self.model.generate_story(
                                pixel_values[j:j+1], self.gpt_tokenizer, max_length=100
                            )
                            reference_story = story_texts[j]
                            
                            # Calculate BLEU score
                            bleu_score = self.accuracy_metrics.calculate_bleu_score(
                                reference_story, generated_story
                            )
                            total_bleu += bleu_score
                            
                            # Calculate ROUGE scores
                            rouge_scores = self.accuracy_metrics.calculate_rouge_scores(
                                reference_story, generated_story
                            )
                            total_rouge1 += rouge_scores['rouge1']
                            total_rouge2 += rouge_scores['rouge2']
                            total_rougeL += rouge_scores['rougeL']
                            
                            num_generated_samples += 1
                        except Exception as e:
                            logger.warning(f"Error in story generation: {e}")
                
                progress_bar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'val_acc': f"{batch_accuracy:.2f}%"
                })
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_perplexity = self.accuracy_metrics.calculate_perplexity(avg_loss)
        
        # Calculate generation metrics
        if num_generated_samples > 0:
            avg_bleu = total_bleu / num_generated_samples
            avg_rouge1 = total_rouge1 / num_generated_samples
            avg_rouge2 = total_rouge2 / num_generated_samples
            avg_rougeL = total_rougeL / num_generated_samples
        else:
            avg_bleu = avg_rouge1 = avg_rouge2 = avg_rougeL = 0.0
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_accuracy)
        self.val_bleu_scores.append(avg_bleu)
        self.val_rouge_scores.append({
            'rouge1': avg_rouge1,
            'rouge2': avg_rouge2,
            'rougeL': avg_rougeL
        })
        self.val_perplexities.append(avg_perplexity)
        
        return avg_loss, avg_accuracy, avg_bleu, avg_rouge1, avg_perplexity
    
    def save_model(self, epoch, val_loss, accuracy, is_best=False):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'accuracy': accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'val_bleu_scores': self.val_bleu_scores,
            'val_rouge_scores': self.val_rouge_scores,
            'val_perplexities': self.val_perplexities,
        }
        
        if is_best:
            best_path = os.path.join(self.save_dir, "models", "best_model.pth")
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}, accuracy: {accuracy:.2f}%")
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, "models", f"checkpoint_epoch_{epoch+1}_{timestamp}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path
    
    def save_metrics(self):
        """Save training metrics"""
        # Prepare ROUGE scores for JSON serialization
        rouge_scores_serializable = []
        for score in self.val_rouge_scores:
            rouge_scores_serializable.append({
                'rouge1': float(score['rouge1']),
                'rouge2': float(score['rouge2']),
                'rougeL': float(score['rougeL'])
            })
        
        metrics = {
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'train_accuracies': [float(x) for x in self.train_accuracies],
            'val_accuracies': [float(x) for x in self.val_accuracies],
            'val_bleu_scores': [float(x) for x in self.val_bleu_scores],
            'val_rouge_scores': rouge_scores_serializable,
            'val_perplexities': [float(x) for x in self.val_perplexities],
            'best_val_loss': float(self.best_val_loss),
            'best_accuracy': float(self.best_accuracy),
            'best_model_path': self.best_model_path
        }
        
        metrics_path = os.path.join(self.save_dir, "metrics", "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save as CSV for easy analysis
        df_metrics = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_accuracy': self.train_accuracies,
            'val_accuracy': self.val_accuracies,
            'val_bleu_score': self.val_bleu_scores,
            'val_perplexity': self.val_perplexities
        })
        
        # Add ROUGE scores
        for i, rouge_score in enumerate(self.val_rouge_scores):
            df_metrics.loc[i, 'val_rouge1'] = rouge_score['rouge1']
            df_metrics.loc[i, 'val_rouge2'] = rouge_score['rouge2']
            df_metrics.loc[i, 'val_rougeL'] = rouge_score['rougeL']
        
        csv_path = os.path.join(self.save_dir, "metrics", "training_metrics.csv")
        df_metrics.to_csv(csv_path, index=False)
        
    def plot_metrics(self):
        """Plot and save training metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_accuracies, label='Training Accuracy', color='blue')
        axes[0, 1].plot(self.val_accuracies, label='Validation Accuracy', color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # BLEU score plot
        axes[0, 2].plot(self.val_bleu_scores, label='BLEU Score', color='green')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('BLEU Score (%)')
        axes[0, 2].set_title('Validation BLEU Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # ROUGE scores plot
        if self.val_rouge_scores:
            rouge1_scores = [score['rouge1'] for score in self.val_rouge_scores]
            rouge2_scores = [score['rouge2'] for score in self.val_rouge_scores]
            rougeL_scores = [score['rougeL'] for score in self.val_rouge_scores]
            
            axes[1, 0].plot(rouge1_scores, label='ROUGE-1', color='purple')
            axes[1, 0].plot(rouge2_scores, label='ROUGE-2', color='orange')
            axes[1, 0].plot(rougeL_scores, label='ROUGE-L', color='brown')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('ROUGE Score (%)')
            axes[1, 0].set_title('Validation ROUGE Scores')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Perplexity plot
        axes[1, 1].plot(self.val_perplexities, label='Perplexity', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Perplexity')
        axes[1, 1].set_title('Validation Perplexity')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')  # Log scale for perplexity
        
        # Learning rate plot (if available)
        if hasattr(self.scheduler, 'get_last_lr'):
            lrs = [self.scheduler.get_last_lr()[0] for _ in range(len(self.train_losses))]
            axes[1, 2].plot(lrs, label='Learning Rate', color='green')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_title('Learning Rate Schedule')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "plots", "training_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def print_metrics_summary(self, epoch, train_loss, train_acc, val_loss, val_acc, bleu_score, rouge1_score, perplexity):
        """Print a comprehensive metrics summary"""
        print("\n" + "="*80)
        print(f"EPOCH {epoch+1} SUMMARY")
        print("="*80)
        print(f"Training Loss:      {train_loss:.4f}")
        print(f"Training Accuracy:  {train_acc:.2f}%")
        print(f"Validation Loss:    {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"BLEU Score:         {bleu_score:.2f}%")
        print(f"ROUGE-1 Score:      {rouge1_score:.2f}%")
        print(f"Perplexity:         {perplexity:.2f}")
        
        # Show improvement indicators
        if len(self.val_accuracies) > 1:
            acc_change = val_acc - self.val_accuracies[-2]
            loss_change = val_loss - self.val_losses[-2]
            print(f"Accuracy Change:    {acc_change:+.2f}% {'📈' if acc_change > 0 else '📉' if acc_change < 0 else '➡️'}")
            print(f"Loss Change:        {loss_change:+.4f} {'📉' if loss_change < 0 else '📈' if loss_change > 0 else '➡️'}")
        
        if val_acc > self.best_accuracy:
            print("🏆 NEW BEST ACCURACY!")
        if val_loss < self.best_val_loss:
            print("🏆 NEW BEST LOSS!")
        print("="*80)
        
    def train(self, num_epochs, save_every=5):
        """Main training loop with comprehensive metrics"""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        print("\n🚀 Starting Image-to-Story Training with Accuracy Metrics!")
        print(f"📊 Metrics tracked: Loss, Token Accuracy, BLEU, ROUGE-1/2/L, Perplexity")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, bleu_score, rouge1_score, perplexity = self.validate(epoch)
            
            # Update best metrics
            is_best_loss = val_loss < self.best_val_loss
            is_best_acc = val_acc > self.best_accuracy
            
            if is_best_loss:
                self.best_val_loss = val_loss
            if is_best_acc:
                self.best_accuracy = val_acc
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, BLEU: {bleu_score:.2f}%")
            
            # Print comprehensive summary
            self.print_metrics_summary(epoch, train_loss, train_acc, val_loss, val_acc, bleu_score, rouge1_score, perplexity)
            
            # Save best model
            if is_best_loss or is_best_acc:
                self.save_model(epoch, val_loss, val_acc, True)
            
            # Save model checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(epoch, val_loss, val_acc, False)
            
            # Save metrics and plots
            self.save_metrics()
            self.plot_metrics()
        
        # Final summary
        print("\n🎉 TRAINING COMPLETED!")
        print("="*80)
        print("FINAL RESULTS:")
        print(f"Best Validation Loss:     {self.best_val_loss:.4f}")
        print(f"Best Validation Accuracy: {self.best_accuracy:.2f}%")
        print(f"Final BLEU Score:         {self.val_bleu_scores[-1]:.2f}%")
        print(f"Final ROUGE-1 Score:      {self.val_rouge_scores[-1]['rouge1']:.2f}%")
        print(f"Final Perplexity:         {self.val_perplexities[-1]:.2f}")
        print(f"Best model saved at:      {self.best_model_path}")
        print("="*80)
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation accuracy: {self.best_accuracy:.2f}%")
        logger.info(f"Best model saved at: {self.best_model_path}")

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Pad story tokens
    story_tokens = [item['story_tokens'] for item in batch]
    story_tokens_padded = pad_sequence(story_tokens, batch_first=True, padding_value=50256)  # GPT2 default pad token
    
    story_texts = [item['story_text'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'story_tokens': story_tokens_padded,
        'story_texts': story_texts
    }

def main():
    # Configuration
    config = {
        'csv_path': r"C:\Users\Murugan\cartoon_story_generator\data\raw_images\comic text dataset.csv",  # UPDATE THIS PATH
        'image_dir': r"C:\Users\Murugan\cartoon_story_generator\data\converted_images",      # UPDATE THIS PATH
        'val_csv_path': 'path/to/your/val_dataset.csv',  # Optional separate validation set
        'batch_size': 8,
        'num_epochs': 20,
        'learning_rate': 5e-5,
        'max_story_length': 512,
        'save_dir': 'blip model/training_output',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'val_split': 0.2,  # If no separate validation set provided
    }
    
    # Check if paths are updated
    if config['csv_path'] == 'path/to/your/dataset.csv' or config['image_dir'] == 'path/to/your/images':
        print("\n" + "="*60)
        print("ERROR: Please update the file paths in the config!")
        print("="*60)
        print("You need to update these paths in the main() function:")
        print(f"- csv_path: Currently set to '{config['csv_path']}'")
        print(f"- image_dir: Currently set to '{config['image_dir']}'")
        print("\nExample:")
        print("config = {")
        print("    'csv_path': 'C:/Users/Murugan/cartoon_story_generator/dataset.csv',")
        print("    'image_dir': 'C:/Users/Murugan/cartoon_story_generator/images',")
        print("    # ... rest of config")
        print("}")
        print("="*60)
        return
    
    # Check if files exist
    if not os.path.exists(config['csv_path']):
        print(f"ERROR: CSV file not found: {config['csv_path']}")
        return
    
    if not os.path.exists(config['image_dir']):
        print(f"ERROR: Image directory not found: {config['image_dir']}")
        return
    
    print("=" * 60)
    print("IMAGE-TO-STORY GENERATOR TRAINING WITH ACCURACY METRICS")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Max story length: {config['max_story_length']}")
    print("📊 Metrics: Loss, Accuracy, BLEU, ROUGE, Perplexity")
    print("=" * 60)
    
    # Install required packages
    print("Installing required packages for metrics...")
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge-score", "nltk"])
        print("✅ Required packages installed successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not install packages automatically: {e}")
        print("Please install manually: pip install rouge-score nltk")
    
    # Initialize processors and tokenizers
    print("Loading BLIP processor and GPT tokenizer...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Create datasets
    print("Creating datasets...")
    full_dataset = ImageStoryDataset(
        config['csv_path'], 
        config['image_dir'], 
        blip_processor, 
        gpt_tokenizer,
        config['max_story_length']
    )
    
    # Split dataset or use separate validation set
    if 'val_csv_path' in config and os.path.exists(config['val_csv_path']):
        val_dataset = ImageStoryDataset(
            config['val_csv_path'], 
            config['image_dir'], 
            blip_processor, 
            gpt_tokenizer,
            config['max_story_length']
        )
        train_dataset = full_dataset
    else:
        # Split the dataset
        train_size = int((1 - config['val_split']) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize model
    print("Initializing model...")
    model = ImageToStoryModel()
    
    # Resize GPT embeddings to accommodate new tokens
    model.gpt.resize_token_embeddings(len(gpt_tokenizer))
    
    model.to(config['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = ImageStoryTrainer(model, train_loader, val_loader, gpt_tokenizer, config['device'], config['save_dir'])
    
    # Setup optimizer
    total_steps = len(train_loader) * config['num_epochs']
    warmup_steps = int(0.1 * total_steps)
    trainer.setup_optimizer(
        learning_rate=config['learning_rate'],
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    
    # Start training
    print("Starting training with comprehensive accuracy metrics...")
    trainer.train(config['num_epochs'])
    
    print("Training completed successfully!")
    print(f"Results saved in: {config['save_dir']}")
    print(f"📊 Check '{config['save_dir']}/metrics/' for detailed accuracy metrics")
    print(f"📈 Check '{config['save_dir']}/plots/' for training visualizations")

if __name__ == "__main__":
    main()