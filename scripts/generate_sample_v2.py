#!/usr/bin/env python3
"""
Generate 'crushed_rock' particles with a single size value (in [0.4..20] mm, truncated normal)
and 2D positions (x,y) on a 300×300 mm table.

Usage Example:
  python generate_sample.py --samplesize 200 --size-mean 5.0 --size-sigma 2.0 -o particles.json
"""

import sys
import json
import click
import numpy as np
from datetime import datetime, timezone
from scipy.stats import truncnorm

@click.command()
@click.option('--samplesize', '-n', default=200, type=int,
              help='Number of particles to generate.')
@click.option('--output', '-o', default='particles.json',
              help='JSON output file path.')
@click.option('--shape-type', default='crushed_rock',
              help='Particle geometry type.')
@click.option('--size-mean', default=5.0, type=float,
              help='Mean particle size (mm).')
@click.option('--size-sigma', default=2.0, type=float,
              help='Std dev for particle size (mm).')
def generate_sample(samplesize, output, shape_type, size_mean, size_sigma):
    """
    Generate a set of 'crushed_rock' particles with a single uniform size and 2D (x,y) positions,
    placed randomly on a 300×300 mm table.
    
    The final JSON will contain, for each particle:
      - "size": the uniform scaling value (mm)
      - "x", "y": positions in mm (fully contained within a margin inside the table)
    
    This JSON can be used 1:1 in Blender to position and uniformly scale your rock model.
    """
    # Table is 300×300 mm.
    table_size = 300

    # 1) Generate sizes using a truncated normal distribution in [0.1, 14] mm.
    min_size = 0.1
    max_size = 15

    def get_truncnorm(mean, sigma, lower, upper, n):
        a, b = (lower - mean) / sigma, (upper - mean) / sigma
        return truncnorm(a, b, loc=mean, scale=sigma).rvs(n)

    sizes = get_truncnorm(mean=size_mean, sigma=size_sigma,
                          lower=min_size, upper=max_size, n=samplesize)

    # 2) Generate x and y coordinates.
    # We'll place particles within a margin to avoid being too close to the table edges.
    # For a 300 mm table (extending from -150 to +150 mm), we use a margin fraction,

    half_range = (table_size / 2.0) - 20

    x_coords = np.random.uniform(low=-half_range, high=half_range, size=samplesize)
    y_coords = np.random.uniform(low=-half_range, high=half_range, size=samplesize)

    # 3) Build JSON data
    data = {
        "shape_type": shape_type,
        "size_mean": size_mean,
        "size_sigma": size_sigma,
        "table_size": table_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samplesize": samplesize,
        "particles": []
    }

    for i in range(samplesize):
        # We don't need a z coordinate; positions are 2D.
        particle = {
            "size": float(sizes[i]),
            "x": float(x_coords[i]),
            "y": float(y_coords[i])
        }
        data["particles"].append(particle)

    # 4) Write JSON file.
    try:
        with open(output, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved {samplesize} particles to {output}")
        print(f"Table size (mm): {table_size}, x,y in roughly [-{half_range}, +{half_range}]")
    except Exception as e:
        print(f"Error saving file {output}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_sample()
