import os
import json
import pandas as pd
import numpy as np

# Root folder containing the dataset
root_folder = "data/dataset"
output_csv = "data/granules_dataset.csv"

def calculate_percentiles(size_list):
    """Helper function to calculate size distribution percentiles."""
    return {
        "d10": np.percentile(size_list, 10),
        "d50": np.percentile(size_list, 50),
        "d90": np.percentile(size_list, 90)
    }

# Define the particles and renders folders
particles_folder = os.path.join(root_folder, "particles")
renders_folder = os.path.join(root_folder, "renders")

# Check if both folders exist
if not os.path.exists(particles_folder) or not os.path.exists(renders_folder):
    print(f"Missing 'particles' or 'renders' folder in {root_folder}.")
    exit()

dataset = []
for json_file in os.listdir(particles_folder):
    if json_file.endswith(".json"):
        # Expecting format "particles_XXXX.json"
        base_name = os.path.basename(json_file)
        try:
            number = base_name.split('_')[1].split('.')[0]
        except IndexError:
            print(f"Unexpected file name format: {base_name}")
            continue

        # Construct corresponding image file name: "render_XXXX.png"
        image_file = f"render_{number}.png"
        image_path = os.path.join(renders_folder, image_file)

        # Skip if the image file does not exist
        if not os.path.exists(image_path):
            print(f"Image not found for {json_file}, skipping...")
            continue

        json_path = os.path.join(particles_folder, json_file)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                sizes = [particle["size"] for particle in data.get("particles", [])]
                percentiles = calculate_percentiles(sizes)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {json_path}: {e}, skipping...")
            continue

        # Add the record to the dataset
        dataset.append({
            "image_path": image_path,
            **percentiles
        })

# Save the dataset to a CSV file
df = pd.DataFrame(dataset)
df.to_csv(output_csv, index=False)

print(f"Dataset created and saved to {output_csv}")
