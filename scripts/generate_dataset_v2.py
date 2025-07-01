#!/usr/bin/env python3
"""
Generate a large dataset (e.g. 2000 images + JSONs) of synthetic crushed‐rock renders,
varying parameters like size range and table size. The sample generator now produces
JSON files with only a single size value and 2D (x,y) positions (table_size is fixed at 300mm).

Requirements:
- generate_sample.py: updated script that uses --size-mean, --size-sigma, and --samplesize
                       to output particles with keys "size", "x", "y" (table_size fixed at 14 cm).
- blender_powder.py: your Blender rendering script that reads absolute mm coordinates.
- Blender 4.x installed and callable as "blender"
- (Optional) GPU-capable environment for faster renders.
"""

import os
import random
import subprocess
import time
import re
import argparse

DEFAULT_NUM_SAMPLES = 2000
DEFAULT_SIZE_MEAN_RANGE = (8.0, 12.0)
DEFAULT_SIZE_SIGMA_RANGE = (6.0, 8.0) 
DEFAULT_PARTICLE_COUNT_RANGE = (800, 1000)

GENERATE_SCRIPT = "./generate_sample_v2.py"
BLENDER_SCRIPT  = "./blender_powder_v2.py"

JSON_OUTPUT_DIR = "../data/dataset/particles"
IMAGE_OUTPUT_DIR = "../data/dataset/renders"

os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

file_pattern = re.compile(r"particles_(\d{4})\.json")

def find_last_index():
    """Find the highest existing index in the dataset folder to continue numbering."""
    existing_files = os.listdir(JSON_OUTPUT_DIR)
    indices = []
    
    for filename in existing_files:
        match = file_pattern.match(filename)
        if match:
            indices.append(int(match.group(1)))

    return max(indices) if indices else 0  # Start from 1 if no files exist

def main():
    """Loop over NUM_SAMPLES to produce each JSON & PNG."""
    
    parser = argparse.ArgumentParser(description='Generate synthetic granular material dataset')
    parser.add_argument('--num_samples', type=int, default=DEFAULT_NUM_SAMPLES,
                       help=f'Number of samples to generate (default: {DEFAULT_NUM_SAMPLES})')
    parser.add_argument('--size_mean_min', type=float, default=DEFAULT_SIZE_MEAN_RANGE[0],
                       help=f'Minimum size mean (default: {DEFAULT_SIZE_MEAN_RANGE[0]})')
    parser.add_argument('--size_mean_max', type=float, default=DEFAULT_SIZE_MEAN_RANGE[1],
                       help=f'Maximum size mean (default: {DEFAULT_SIZE_MEAN_RANGE[1]})')
    parser.add_argument('--size_sigma_min', type=float, default=DEFAULT_SIZE_SIGMA_RANGE[0],
                       help=f'Minimum size sigma (default: {DEFAULT_SIZE_SIGMA_RANGE[0]})')
    parser.add_argument('--size_sigma_max', type=float, default=DEFAULT_SIZE_SIGMA_RANGE[1],
                       help=f'Maximum size sigma (default: {DEFAULT_SIZE_SIGMA_RANGE[1]})')
    parser.add_argument('--particles_min', type=int, default=DEFAULT_PARTICLE_COUNT_RANGE[0],
                       help=f'Minimum particles per sample (default: {DEFAULT_PARTICLE_COUNT_RANGE[0]})')
    parser.add_argument('--particles_max', type=int, default=DEFAULT_PARTICLE_COUNT_RANGE[1],
                       help=f'Maximum particles per sample (default: {DEFAULT_PARTICLE_COUNT_RANGE[1]})')
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_samples} samples with:")
    print(f"  Size mean range: [{args.size_mean_min}, {args.size_mean_max}]")
    print(f"  Size sigma range: [{args.size_sigma_min}, {args.size_sigma_max}]")
    print(f"  Particles per sample: [{args.particles_min}, {args.particles_max}]")
    print()

    start_index = find_last_index() + 1
    end_index = start_index + args.num_samples

    for i in range(start_index, end_index):
        size_mean = random.uniform(args.size_mean_min, args.size_mean_max)
        size_sigma = random.uniform(args.size_sigma_min, args.size_sigma_max)
        samplesize = random.randint(args.particles_min, args.particles_max)
        
        json_name  = f"particles_{i:04d}.json"
        image_name = f"render_{i:04d}.png"
        
        json_path  = os.path.join(JSON_OUTPUT_DIR, json_name)
        image_path = os.path.join(IMAGE_OUTPUT_DIR, image_name)
        
        env = os.environ.copy()
        env["PARTICLESPATH"]  = os.path.abspath(json_path)
        env["RENDERPATH"]     = os.path.abspath(image_path)

        cmd_generate = [
            "python", GENERATE_SCRIPT,
            "--size-mean", str(size_mean),
            "--size-sigma", str(size_sigma),
            "--samplesize", str(samplesize),
            "-o", json_path
        ]
        
        print(f"[{i - start_index + 1}/{args.num_samples}] Generating JSON via:", " ".join(cmd_generate))
        subprocess.run(cmd_generate, check=True)

        env = os.environ.copy()
        env["PARTICLESPATH"] = os.path.abspath(json_path)
        env["RENDERPATH"]    = os.path.abspath(image_path)
        
        cmd_blender = [
            "blender", "--background", "--python", os.path.abspath(BLENDER_SCRIPT)
        ]
        
        print(f"[{i - start_index + 1}/{args.num_samples}] Rendering image via:", " ".join(cmd_blender))
        start_time = time.time()
        subprocess.run(cmd_blender, env=env, check=True)
        elapsed = time.time() - start_time
        
        print(f" -> Completed sample {i - start_index + 1}/{args.num_samples} in {elapsed:.1f}s. "
              f"Saved: {json_name}, {image_name}\n")

if __name__ == "__main__":
    main()
