import os
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


def augment_polygon_dataset(input_dir, output_dir, num_augmentations=3):
    """
    Augment the polygon dataset with rotations and scaling
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define augmentation transforms
    augment_transforms = [
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ]
    
    # Process training data
    with open('training/data.json', 'r') as f:
        train_data = json.load(f)
    
    augmented_data = []
    augmented_inputs = []
    augmented_outputs = []
    
    print("Generating augmented data...")
    
    for item_idx, item in enumerate(tqdm(train_data)):
        input_path = os.path.join('training/inputs', item['input_polygon'])
        output_path = os.path.join('training/outputs', item['output_image'])
        
        # Load original images
        input_img = Image.open(input_path).convert('RGB')
        output_img = Image.open(output_path).convert('RGB')
        
        # Save original
        augmented_inputs.append(item['input_polygon'])
        augmented_outputs.append(item['output_image'])
        augmented_data.append(item)
        
        # Generate augmentations
        for aug_idx in range(num_augmentations):
            # Randomly select transforms
            selected_transforms = np.random.choice(augment_transforms, 
                                                   size=np.random.randint(1, 4), 
                                                   replace=False)
            
            transform = transforms.Compose([
                *selected_transforms,
                transforms.Resize((256, 256))
            ])
            
            # Apply transforms
            aug_input = transform(input_img)
            aug_output = transform(output_img)
            
            # Save augmented images
            input_filename = f"aug_{aug_idx}_{item['input_polygon']}"
            output_filename = f"aug_{aug_idx}_{item['output_image']}"
            
            aug_input.save(os.path.join(output_dir, 'inputs', input_filename))
            aug_output.save(os.path.join(output_dir, 'outputs', output_filename))
            
            # Add to data
            augmented_data.append({
                "input_polygon": input_filename,
                "colour": item['colour'],
                "output_image": output_filename
            })
    
    # Save augmented data
    with open(os.path.join(output_dir, 'data.json'), 'w') as f:
        json.dump(augmented_data, f, indent=2)
    
    print(f"Generated {len(augmented_data)} total samples")
    print(f"Original: {len(train_data)} samples")
    print(f"Augmented: {len(augmented_data) - len(train_data)} samples")


def create_synthetic_polygons():
    """
    Generate synthetic polygon images for additional training data
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    synthetic_dir = 'synthetic_data'
    os.makedirs(f'{synthetic_dir}/inputs', exist_ok=True)
    os.makedirs(f'{synthetic_dir}/outputs', exist_ok=True)
    
    colors = {
        'red': (1, 0, 0),
        'blue': (0, 0, 1),
        'green': (0, 1, 0),
        'yellow': (1, 1, 0),
        'purple': (0.5, 0, 0.5),
        'orange': (1, 0.5, 0),
        'cyan': (0, 1, 1),
        'magenta': (1, 0, 1)
    }
    
    polygons = [
        ('triangle', [(0.5, 0.1), (0.1, 0.9), (0.9, 0.9)]),
        ('square', [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]),
        ('pentagon', [(0.5, 0.1), (0.1, 0.4), (0.3, 0.9), (0.7, 0.9), (0.9, 0.4)]),
        ('hexagon', [(0.5, 0.1), (0.1, 0.3), (0.1, 0.7), (0.5, 0.9), (0.9, 0.7), (0.9, 0.3)])
    ]
    
    synthetic_data = []
    
    for poly_name, vertices in polygons:
        for color_name, color_rgb in colors.items():
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(2.56, 2.56), dpi=100)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Create outline (input)
            outline = patches.Polygon(vertices, fill=False, edgecolor='black', linewidth=3)
            ax.add_patch(outline)
            ax.set_facecolor('white')
            
            input_filename = f"{poly_name}_{color_name}_input.png"
            plt.savefig(f'{synthetic_dir}/inputs/{input_filename}', 
                       bbox_inches='tight', pad_inches=0, dpi=100)
            
            # Create filled version (output)
            ax.clear()
            filled = patches.Polygon(vertices, fill=True, facecolor=color_rgb, 
                                   edgecolor=color_rgb, linewidth=1)
            ax.add_patch(filled)
            ax.set_facecolor('white')
            
            output_filename = f"{poly_name}_{color_name}_output.png"
            plt.savefig(f'{synthetic_dir}/outputs/{output_filename}', 
                       bbox_inches='tight', pad_inches=0, dpi=100)
            
            plt.close()
            
            # Add to data
            synthetic_data.append({
                "input_polygon": input_filename,
                "colour": color_name,
                "output_image": output_filename
            })
    
    # Save synthetic data
    with open(f'{synthetic_dir}/data.json', 'w') as f:
        json.dump(synthetic_data, f, indent=2)
    
    print(f"Generated {len(synthetic_data)} synthetic samples")


if __name__ == "__main__":
    # Augment existing training data
    augment_polygon_dataset('training', 'augmented_training', num_augmentations=2)
    
    # Generate synthetic data
    create_synthetic_polygons()
    
    print("Data augmentation complete!")