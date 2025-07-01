#!/usr/bin/env python3
"""
Generate CSV dataset from existing particle JSON files and render images.
This script reads particle size data from JSON files and creates a CSV with
image paths and corresponding d10, d50, d90 values.
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def calculate_percentiles(size_list):
    """Calculate size distribution percentiles (d10, d50, d90)."""
    if not size_list:
        return {"d10": 0, "d50": 0, "d90": 0}
    
    return {
        "d10": np.percentile(size_list, 10),
        "d50": np.percentile(size_list, 50),
        "d90": np.percentile(size_list, 90)
    }

def generate_csv_dataset(particles_dir, renders_dir, output_csv, relative_paths=True):
    """
    Generate CSV dataset from particles JSON and render images.
    
    Args:
        particles_dir: Directory containing particle JSON files
        renders_dir: Directory containing rendered images
        output_csv: Output CSV file path
        relative_paths: Whether to use relative paths in CSV
    """
    
    # Check if directories exist
    if not os.path.exists(particles_dir):
        raise FileNotFoundError(f"Particles directory not found: {particles_dir}")
    if not os.path.exists(renders_dir):
        raise FileNotFoundError(f"Renders directory not found: {renders_dir}")
    
    dataset = []
    processed_count = 0
    skipped_count = 0
    
    print(f"Processing files from:")
    print(f"  Particles: {particles_dir}")
    print(f"  Renders: {renders_dir}")
    print()
    
    # Process all JSON files in particles directory
    json_files = [f for f in os.listdir(particles_dir) if f.endswith(".json")]
    json_files.sort()  # Process in order
    
    for json_file in json_files:
        # Extract number from filename (e.g., particles_0001.json -> 0001)
        base_name = os.path.basename(json_file)
        try:
            if "particles_" in base_name:
                number = base_name.split('_')[1].split('.')[0]
            else:
                # Try to extract number from filename
                number = ''.join(filter(str.isdigit, base_name))
                if not number:
                    print(f"Cannot extract number from: {base_name}, skipping...")
                    skipped_count += 1
                    continue
        except IndexError:
            print(f"Unexpected file name format: {base_name}, skipping...")
            skipped_count += 1
            continue
        
        # Find corresponding image file
        possible_image_names = [
            f"render_{number}.png",
            f"render_{int(number):04d}.png",
            f"image_{number}.png",
            f"img_{number}.png"
        ]
        
        image_path = None
        for img_name in possible_image_names:
            potential_path = os.path.join(renders_dir, img_name)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            print(f"No matching image found for {json_file}, skipping...")
            skipped_count += 1
            continue
        
        # Read and process JSON file
        json_path = os.path.join(particles_dir, json_file)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                
                # Extract particle sizes
                particles = data.get("particles", [])
                if not particles:
                    print(f"No particles found in {json_file}, skipping...")
                    skipped_count += 1
                    continue
                
                sizes = []
                for particle in particles:
                    if "size" in particle:
                        sizes.append(particle["size"])
                    elif "diameter" in particle:
                        sizes.append(particle["diameter"])
                    elif "radius" in particle:
                        sizes.append(particle["radius"] * 2)  # Convert radius to diameter
                
                if not sizes:
                    print(f"No size data found in {json_file}, skipping...")
                    skipped_count += 1
                    continue
                
                # Calculate percentiles
                percentiles = calculate_percentiles(sizes)
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {json_path}: {e}, skipping...")
            skipped_count += 1
            continue
        
        # Determine image path format for CSV
        if relative_paths:
            # Make path relative to current working directory
            final_image_path = os.path.relpath(image_path)
        else:
            final_image_path = os.path.abspath(image_path)
        
        # Add record to dataset
        dataset.append({
            "image_path": final_image_path,
            "d10": percentiles["d10"],
            "d50": percentiles["d50"],
            "d90": percentiles["d90"]
        })
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} samples...")
    
    # Create DataFrame and save to CSV
    if not dataset:
        raise ValueError("No valid data found to create CSV dataset")
    
    df = pd.DataFrame(dataset)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\nDataset generation complete!")
    print(f"  Processed: {processed_count} samples")
    print(f"  Skipped: {skipped_count} samples")
    print(f"  Output: {output_csv}")
    print(f"\nDataset summary:")
    print(f"  d10 range: [{df['d10'].min():.3f}, {df['d10'].max():.3f}]")
    print(f"  d50 range: [{df['d50'].min():.3f}, {df['d50'].max():.3f}]")
    print(f"  d90 range: [{df['d90'].min():.3f}, {df['d90'].max():.3f}]")

def main():
    parser = argparse.ArgumentParser(
        description='Generate CSV dataset from particle JSON files and render images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_csv.py
  python generate_csv.py --particles_dir data/dataset/particles --renders_dir data/dataset/renders
  python generate_csv.py --output data/my_dataset.csv --absolute_paths
        """
    )
    
    parser.add_argument(
        '--particles_dir', 
        type=str, 
        default='data/dataset/particles',
        help='Directory containing particle JSON files (default: data/dataset/particles)'
    )
    
    parser.add_argument(
        '--renders_dir', 
        type=str, 
        default='data/dataset/renders',
        help='Directory containing rendered images (default: data/dataset/renders)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='data/granules_dataset.csv',
        help='Output CSV file path (default: data/granules_dataset.csv)'
    )
    
    parser.add_argument(
        '--absolute_paths', 
        action='store_true',
        help='Use absolute paths in CSV instead of relative paths'
    )
    
    args = parser.parse_args()
    
    try:
        generate_csv_dataset(
            particles_dir=args.particles_dir,
            renders_dir=args.renders_dir,
            output_csv=args.output,
            relative_paths=not args.absolute_paths
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 