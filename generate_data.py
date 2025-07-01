#!/usr/bin/env python3
"""
Main data generation script for creating synthetic granular material datasets.

This script provides a clean interface for generating synthetic particle configurations
and rendering them using Blender.
"""

import argparse
import os
import subprocess
import sys


def main():
    """Main entry point for data generation."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic granular material dataset using Blender',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=2000,
        help='Number of synthetic samples to generate'
    )
    
    parser.add_argument(
        '--size_mean_min', 
        type=float, 
        default=8.0,
        help='Minimum value for particle size mean'
    )
    
    parser.add_argument(
        '--size_mean_max', 
        type=float, 
        default=12.0,
        help='Maximum value for particle size mean'
    )
    
    parser.add_argument(
        '--size_sigma_min', 
        type=float, 
        default=6.0,
        help='Minimum value for particle size standard deviation'
    )
    
    parser.add_argument(
        '--size_sigma_max', 
        type=float, 
        default=8.0,
        help='Maximum value for particle size standard deviation'
    )
    
    parser.add_argument(
        '--particles_min', 
        type=int, 
        default=800,
        help='Minimum number of particles per sample'
    )
    
    parser.add_argument(
        '--particles_max', 
        type=int, 
        default=1000,
        help='Maximum number of particles per sample'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/dataset',
        help='Directory to store generated dataset'
    )
    
    parser.add_argument(
        '--check_blender', 
        action='store_true',
        help='Check if Blender is available before starting generation'
    )
    
    args = parser.parse_args()
    
    if args.check_blender:
        try:
            result = subprocess.run(['blender', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✓ Blender found: {result.stdout.split()[1]}")
            else:
                print("✗ Blender not found or not working properly")
                sys.exit(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("✗ Blender not found in PATH")
            print("Please install Blender and ensure it's accessible via command line")
            sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting data generation with:")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Size mean range: [{args.size_mean_min}, {args.size_mean_max}]")
    print(f"  Size sigma range: [{args.size_sigma_min}, {args.size_sigma_max}]")
    print(f"  Particles per sample: [{args.particles_min}, {args.particles_max}]")
    print(f"  Output directory: {args.output_dir}")
    print("-" * 50)
    
    scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
    original_dir = os.getcwd()
    
    try:
        os.chdir(scripts_dir)
        
        cmd = [
            sys.executable, 'generate_dataset_v2.py',
            '--num_samples', str(args.num_samples),
            '--size_mean_min', str(args.size_mean_min),
            '--size_mean_max', str(args.size_mean_max),
            '--size_sigma_min', str(args.size_sigma_min),
            '--size_sigma_max', str(args.size_sigma_max),
            '--particles_min', str(args.particles_min),
            '--particles_max', str(args.particles_max)
        ]
        
        print("Running data generation...")
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n✓ Data generation completed successfully!")
            print(f"Dataset saved to: {os.path.abspath(args.output_dir)}")
        else:
            print("\n✗ Data generation failed!")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print("\n⚠ Data generation interrupted by user")
        sys.exit(1)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main() 