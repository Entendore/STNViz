import os
import cv2
import numpy as np
import random
import argparse
import math

def apply_random_affine(img, strong=False):
    """Applies a random affine transformation (Translation, Rotation, Scale, Shear)."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    
    # Random parameters
    angle = random.uniform(-45, 45) if strong else random.uniform(-15, 15)
    scale = random.uniform(0.6, 1.2) if strong else random.uniform(0.8, 1.1)
    tx = random.uniform(-15, 15)
    ty = random.uniform(-15, 15)
    
    # Shear components
    shx = random.uniform(-0.2, 0.2) if strong else 0
    shy = random.uniform(-0.2, 0.2) if strong else 0
    
    # Build 2x3 affine matrix manually for shear + rotation + scale + translation
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    
    # Center, scale, shear
    M = np.float32([
        [cos_a * scale, -sin_a * scale + shx, cx - (cos_a * scale * cx - sin_a * scale * cy + shx * cy) + tx],
        [sin_a * scale + shy, cos_a * scale, cy - (sin_a * scale * cx + cos_a * scale * cy + shy * cx) + ty]
    ])
    
    border_val = (0, 0, 0) if len(img.shape) == 3 else 0
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

def draw_digit(canvas, digit):
    """Draws a digit on the canvas."""
    color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, str(digit), (16, 48), font, 1.6, color, 4, cv2.LINE_AA)

def draw_shape(canvas, shape_type):
    """Draws a geometric shape on the canvas."""
    color = (255, 255, 255)
    if shape_type == 'circle':
        cv2.circle(canvas, (32, 32), 20, color, -1)
    elif shape_type == 'square':
        cv2.rectangle(canvas, (12, 12), (52, 52), color, -1)
    elif shape_type == 'triangle':
        pts = np.array([[32, 10], [54, 50], [10, 50]], np.int32)
        cv2.fillPoly(canvas, [pts], color)
    elif shape_type == 'plus':
        cv2.rectangle(canvas, (22, 12), (42, 52), color, -1)
        cv2.rectangle(canvas, (12, 22), (52, 42), color, -1)

def generate_clutter(canvas):
    """Adds random background noise/lines to make classification harder."""
    for _ in range(random.randint(3, 7)):
        pt1 = (random.randint(0, 64), random.randint(0, 64))
        pt2 = (random.randint(0, 64), random.randint(0, 64))
        cv2.line(canvas, pt1, pt2, (100, 100, 100), 1, cv2.LINE_AA)
    return canvas

def generate_distorted_digits_dataset(root_dir, num_per_class=200, clutter=False):
    """Generates a dataset of heavily distorted digits (0-9)."""
    dataset_name = "cluttered_digits" if clutter else "distorted_digits"
    save_dir = os.path.join(root_dir, dataset_name)
    print(f"Generating {dataset_name} dataset at: {save_dir}")
    
    for digit in range(10):
        class_dir = os.path.join(save_dir, str(digit))
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(num_per_class):
            # 1. Create base image
            canvas = np.zeros((64, 64, 3), dtype=np.uint8)
            
            # 2. Add clutter (STN should learn to ignore this and focus on the digit)
            if clutter:
                canvas = generate_clutter(canvas)
                
            # 3. Draw the target
            draw_digit(canvas, digit)
            
            # 4. Apply strong spatial transformation
            distorted = apply_random_affine(canvas, strong=True)
            
            # 5. Save
            cv2.imwrite(os.path.join(class_dir, f"img_{i}.png"), distorted)
            
    print(f"Finished generating {dataset_name}.")

def generate_shapes_dataset(root_dir, num_per_class=200):
    """Generates a dataset of geometric shapes with varying transformations."""
    save_dir = os.path.join(root_dir, "distorted_shapes")
    print(f"Generating distorted_shapes dataset at: {save_dir}")
    
    shapes = ['circle', 'square', 'triangle', 'plus']
    
    for shape in shapes:
        class_dir = os.path.join(save_dir, shape)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(num_per_class):
            canvas = np.zeros((64, 64, 3), dtype=np.uint8)
            draw_shape(canvas, shape)
            
            # Apply moderate spatial transformation
            distorted = apply_random_affine(canvas, strong=False)
            
            cv2.imwrite(os.path.join(class_dir, f"img_{i}.png"), distorted)
            
    print("Finished generating distorted_shapes.")

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Datasets for Spatial Transformer Networks (STN)")
    parser.add_argument("--output", type=str, default="./stn_datasets", help="Output directory for datasets")
    parser.add_argument("--num", type=int, default=200, help="Number of images per class")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Generate Distorted Digits (Translation + Rotation + Scale + Shear)
    # This is the classic STN test case.
    generate_distorted_digits_dataset(args.output, num_per_class=args.num, clutter=False)
    
    # 2. Generate Cluttered Digits (Background noise + Distortion)
    # This tests the STN's ability to attend to the relevant feature and ignore background noise.
    generate_distorted_digits_dataset(args.output, num_per_class=args.num, clutter=True)
    
    # 3. Generate Distorted Shapes (Simpler task, good for debugging)
    generate_shapes_dataset(args.output, num_per_class=args.num)
    
    print("\nAll datasets generated successfully!")
    print(f"You can now load '{args.output}' subfolders into the STN Training App using 'Custom ImageFolder'.")

if __name__ == "__main__":
    main()