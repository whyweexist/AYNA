import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from unet_model import ConditionalUNet


class PolygonDataset(Dataset):
    def __init__(self, data_json, input_dir, output_dir, color_map, transform=None):
        with open(data_json, 'r') as f:
            self.data = json.load(f)
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.color_map = color_map
        self.transform = transform
        
        # Image transformations
        self.input_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        self.output_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load input polygon image
        input_path = os.path.join(self.input_dir, item['input_polygon'])
        input_image = Image.open(input_path).convert('RGB')
        input_tensor = self.input_transform(input_image)
        
        # Load output colored polygon image
        output_path = os.path.join(self.output_dir, item['output_image'])
        output_image = Image.open(output_path).convert('RGB')
        output_tensor = self.output_transform(output_image)
        
        # Get color index
        color_name = item['colour']
        color_idx = self.color_map[color_name]
        
        return {
            'input': input_tensor,
            'output': output_tensor,
            'color': torch.tensor(color_idx, dtype=torch.long)
        }


def create_color_map():
    """Create a mapping from color names to indices"""
    colors = [
        'red', 'blue', 'green', 'yellow', 'purple', 
        'orange', 'cyan', 'magenta', 'pink', 'brown'
    ]
    # Filter to only include colors present in dataset
    actual_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    return {color: idx for idx, color in enumerate(actual_colors)}


def train_model():
    # Initialize wandb
    wandb.init(
        project="polygon-color-unet",
        config={
            "learning_rate": 1e-3,
            "batch_size": 16,
            "epochs": 100,
            "model": "ConditionalUNet",
            "image_size": 256,
            "num_colors": 8,
            "embed_dim": 64,
            "optimizer": "Adam",
            "loss_function": "MSE",
            "dataset": "Polygon Coloring"
        }
    )
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create color mapping
    color_map = create_color_map()
    print("Color mapping:", color_map)
    
    # Data paths
    train_json = 'training/data.json'
    val_json = 'validation/data.json'
    train_input_dir = 'training/inputs'
    train_output_dir = 'training/outputs'
    val_input_dir = 'validation/inputs'
    val_output_dir = 'validation/outputs'
    
    # Create datasets
    train_dataset = PolygonDataset(train_json, train_input_dir, train_output_dir, color_map)
    val_dataset = PolygonDataset(val_json, val_input_dir, val_output_dir, color_map)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = ConditionalUNet(
        n_channels=1, 
        n_classes=3, 
        num_colors=len(color_map), 
        embed_dim=wandb.config.embed_dim
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Watch model with wandb
    wandb.watch(model, log="all")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(wandb.config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{wandb.config.epochs} - Training')
        
        for batch in train_pbar:
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            colors = batch['color'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, colors)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{wandb.config.epochs} - Validation')
        
        with torch.no_grad():
            for batch in val_pbar:
                inputs = batch['input'].to(device)
                targets = batch['output'].to(device)
                colors = batch['color'].to(device)
                
                outputs = model(inputs, colors)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Log metrics
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'color_map': color_map
            }, 'best_polygon_unet.pth')
            wandb.save('best_polygon_unet.pth')
            print(f"New best model saved with validation loss: {avg_val_loss:.6f}")
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'color_map': color_map
    }, 'final_polygon_unet.pth')
    wandb.save('final_polygon_unet.pth')
    
    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    train_model()