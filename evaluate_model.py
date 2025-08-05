import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from unet_model import ConditionalUNet
import argparse

class ModelEvaluator:
    def __init__(self, model_path, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        
        # Color mapping
        self.COLOR_MAP = {
            'red': 0, 'blue': 1, 'green': 2, 'yellow': 3,
            'purple': 4, 'orange': 5, 'cyan': 6, 'magenta': 7
        }
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
    def load_model(self, model_path):
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = ConditionalUNet(
            n_channels=1, 
            n_classes=3, 
            num_colors=len(self.COLOR_MAP), 
            embed_dim=64
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def calculate_mse(self, generated, target):
        """Calculate Mean Squared Error"""
        return F.mse_loss(generated, target).item()
    
    def calculate_mae(self, generated, target):
        """Calculate Mean Absolute Error"""
        return F.l1_loss(generated, target).item()
    
    def calculate_psnr(self, generated, target):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(generated, target)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    def evaluate_dataset(self, data_json, input_dir, output_dir):
        """Evaluate model on entire dataset"""
        with open(data_json, 'r') as f:
            data = json.load(f)
        
        results = []
        
        for item in data:
            # Load input
            input_path = os.path.join(input_dir, item['input_polygon'])
            input_img = Image.open(input_path).convert('RGB')
            input_tensor = self.transform(input_img).unsqueeze(0).to(self.device)
            
            # Load target
            target_path = os.path.join(output_dir, item['output_image'])
            target_img = Image.open(target_path).convert('RGB')
            target_tensor = transforms.ToTensor()(target_img).unsqueeze(0).to(self.device)
            
            # Get color index
            color_idx = torch.tensor([self.COLOR_MAP[item['colour']]], dtype=torch.long).to(self.device)
            
            # Generate
            with torch.no_grad():
                generated = self.model(input_tensor, color_idx)
            
            # Calculate metrics
            mse = self.calculate_mse(generated, target_tensor)
            mae = self.calculate_mae(generated, target_tensor)
            psnr = self.calculate_psnr(generated, target_tensor)
            
            results.append({
                'input': item['input_polygon'],
                'color': item['colour'],
                'output': item['output_image'],
                'mse': mse,
                'mae': mae,
                'psnr': psnr
            })
        
        return results
    
    def visualize_samples(self, data_json, input_dir, output_dir, num_samples=5, save_dir='evaluation_results'):
        """Visualize sample predictions"""
        os.makedirs(save_dir, exist_ok=True)
        
        with open(data_json, 'r') as f:
            data = json.load(f)
        
        # Select random samples
        indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
        selected_samples = [data[i] for i in indices]
        
        fig, axes = plt.subplots(len(selected_samples), 3, figsize=(12, 4*len(selected_samples)))
        if len(selected_samples) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, item in enumerate(selected_samples):
            # Load images
            input_path = os.path.join(input_dir, item['input_polygon'])
            target_path = os.path.join(output_dir, item['output_image'])
            
            input_img = Image.open(input_path).convert('RGB')
            target_img = Image.open(target_path).convert('RGB')
            
            # Generate prediction
            input_tensor = self.transform(input_img).unsqueeze(0).to(self.device)
            color_idx = torch.tensor([self.COLOR_MAP[item['colour']]], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                generated = self.model(input_tensor, color_idx)
            
            # Convert to PIL
            generated_np = generated.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            generated_np = np.clip(generated_np, 0, 1)
            generated_img = Image.fromarray((generated_np * 255).astype(np.uint8))
            
            # Plot
            axes[idx, 0].imshow(input_img)
            axes[idx, 0].set_title(f"Input: {item['input_polygon']}")
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(generated_img)
            axes[idx, 1].set_title(f"Generated: {item['colour']}")
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(target_img)
            axes[idx, 2].set_title(f"Target: {item['colour']}")
            axes[idx, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_report(self, results):
        """Generate evaluation report"""
        metrics = {
            'mse': [r['mse'] for r in results],
            'mae': [r['mae'] for r in results],
            'psnr': [r['psnr'] for r in results]
        }
        
        report = {
            'total_samples': len(results),
            'mse_mean': np.mean(metrics['mse']),
            'mse_std': np.std(metrics['mse']),
            'mae_mean': np.mean(metrics['mae']),
            'mae_std': np.std(metrics['mae']),
            'psnr_mean': np.mean(metrics['psnr']),
            'psnr_std': np.std(metrics['psnr']),
            'best_mse': np.min(metrics['mse']),
            'worst_mse': np.max(metrics['mse']),
            'best_psnr': np.max(metrics['psnr']),
            'worst_psnr': np.min(metrics['psnr'])
        }
        
        return report
    
    def plot_color_performance(self, results):
        """Plot performance by color"""
        color_metrics = {}
        for result in results:
            color = result['color']
            if color not in color_metrics:
                color_metrics[color] = {'mse': [], 'psnr': []}
            color_metrics[color]['mse'].append(result['mse'])
            color_metrics[color]['psnr'].append(result['psnr'])
        
        colors = list(color_metrics.keys())
        mse_means = [np.mean(color_metrics[c]['mse']) for c in colors]
        psnr_means = [np.mean(color_metrics[c]['psnr']) for c in colors]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # MSE by color
        ax1.bar(colors, mse_means)
        ax1.set_title('MSE by Color')
        ax1.set_ylabel('MSE')
        ax1.tick_params(axis='x', rotation=45)
        
        # PSNR by color
        ax2.bar(colors, psnr_means)
        ax2.set_title('PSNR by Color')
        ax2.set_ylabel('PSNR (dB)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('evaluation_results/color_performance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    parser = argparse.ArgumentParser(description='Evaluate polygon color UNet model')
    parser.add_argument('--model', default='best_polygon_unet.pth', help='Path to model file')
    parser.add_argument('--data', default='validation', choices=['validation', 'training'], help='Dataset to evaluate')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model, device=args.device)
    
    # Select dataset
    if args.data == 'validation':
        data_json = 'validation/data.json'
        input_dir = 'validation/inputs'
        output_dir = 'validation/outputs'
    else:
        data_json = 'training/data.json'
        input_dir = 'training/inputs'
        output_dir = 'training/outputs'
    
    print(f"Evaluating on {args.data} dataset...")
    
    # Evaluate
    results = evaluator.evaluate_dataset(data_json, input_dir, output_dir)
    
    # Generate report
    report = evaluator.generate_report(results)
    
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"Dataset: {args.data}")
    print(f"Total samples: {report['total_samples']}")
    print(f"\nMetrics:")
    print(f"  MSE: {report['mse_mean']:.4f} ± {report['mse_std']:.4f}")
    print(f"  MAE: {report['mae_mean']:.4f} ± {report['mae_std']:.4f}")
    print(f"  PSNR: {report['psnr_mean']:.2f} ± {report['psnr_std']:.2f} dB")
    print(f"\nExtremes:")
    print(f"  Best MSE: {report['best_mse']:.4f}")
    print(f"  Worst MSE: {report['worst_mse']:.4f}")
    print(f"  Best PSNR: {report['best_psnr']:.2f} dB")
    print(f"  Worst PSNR: {report['worst_psnr']:.2f} dB")
    
    # Save detailed results
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('evaluation_results/report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        evaluator.visualize_samples(data_json, input_dir, output_dir, num_samples=args.samples)
        evaluator.plot_color_performance(results)
        print("✓ Visualizations saved to 'evaluation_results/'")

if __name__ == "__main__":
    main()