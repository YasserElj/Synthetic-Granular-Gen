#!/usr/bin/env python3
"""
Main training script for granular material size prediction models.

This script provides a clean interface for training models with different architectures
on the synthetic granular material dataset.
"""

import argparse
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import main as train_main


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(
        description='Train deep learning models for granular material size prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model_type', 
        type=str, 
        choices=['resnet50', 'efficientnet_b0', 'inception_v3', 'cnn'],
        default='resnet50',
        help='Type of model to train'
    )
    
    parser.add_argument(
        '--csv_file', 
        type=str, 
        default='data/granules_dataset.csv',
        help='Path to the dataset CSV file'
    )
    
    parser.add_argument(
        '--train_split', 
        type=float, 
        default=0.7,
        help='Fraction of data to use for training'
    )
    
    parser.add_argument(
        '--val_split', 
        type=float, 
        default=0.15,
        help='Fraction of data to use for validation'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.001,
        help='Learning rate for optimizer'
    )
    
    parser.add_argument(
        '--results_dir', 
        type=str, 
        default='results',
        help='Directory to save training results'
    )
    
    parser.add_argument(
        '--weights_dir', 
        type=str, 
        default='weights',
        help='Directory to save model weights'
    )
    
    args = parser.parse_args()
    
    sys.argv = [
        'train.py',
        '--model_type', args.model_type,
        '--csv', args.csv_file,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr)
    ]
    
    print(f"Starting training with:")
    print(f"  Model: {args.model_type}")
    print(f"  Dataset: {args.csv_file}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print("-" * 50)
    
    train_main()


if __name__ == "__main__":
    main() 