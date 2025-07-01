import os
import argparse
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

def generate_test_csv_from_predictions(old_pred_csv, out_csv):
    """
    Creates a new test CSV from an old prediction file.
    This ensures we use the same image paths and ground-truth as before,
    ignoring the old predicted values, to produce a new test CSV for re-testing.
    """
    df = pd.read_csv(old_pred_csv)
    for col in ["image_path", "true_d10", "true_d50", "true_d90"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in {old_pred_csv}")

    df_test = df[["image_path", "true_d10", "true_d50", "true_d90"]].copy()
    df_test.rename(columns={
        "true_d10": "d10",
        "true_d50": "d50",
        "true_d90": "d90"
    }, inplace=True)

    df_test.to_csv(out_csv, index=False)
    print(f"Generated new test CSV: {out_csv} from {old_pred_csv}")


class GranuleTestDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        target = np.array([row['d10'], row['d50'], row['d90']], dtype=np.float32)
        
        return image, target, img_path

def get_model(model_type, pretrained=False):
    """
    Return the specified model, optionally with pretrained ImageNet weights.
    If pretrained=True, we load base ImageNet weights, 
    then replace the final layer to output 3 values.
    """
    if model_type == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 3)
    
    elif model_type == "efficientnet_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        if pretrained:
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 3)
    
    elif model_type == "inception_v3":
        from torchvision.models import inception_v3, Inception_V3_Weights
        # Pretrained inception_v3 typically uses aux_logits=True
        if pretrained:
            model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        else:
            model = inception_v3(weights=None, aux_logits=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 3)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

def mean_absolute_error_manual(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error_manual(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def plot_predictions(model, test_loader, device, model_name, save_path):
    """Plots predicted vs. ground-truth for d10, d50, and d90."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images, labels, _ = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple): # Handle InceptionV3 auxiliary output
                outputs = outputs[0]
            
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = ['d10', 'd50', 'd90']
    plt.figure(figsize=(18, 6))
    
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(all_labels[:, i], all_predictions[:, i], alpha=0.7,
                    label=f'Predicted vs True {metrics[i]}')
        plt.plot([all_labels[:, i].min(), all_labels[:, i].max()],
                 [all_labels[:, i].min(), all_labels[:, i].max()], 'r--')
        plt.xlabel(f'True {metrics[i]}')
        plt.ylabel(f'Predicted {metrics[i]}')
        plt.title(f'{metrics[i]}: Prediction vs Ground Truth')
        plt.legend()
        plt.grid(True)
    
    plt.suptitle(f"Model: {model_name} — Predictions vs. Ground Truth", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate test CSV from old predictions (optional), run inference & metrics.")
    
    parser.add_argument("--old_pred_csv", type=str, default=None,
                        help="Existing predictions CSV with columns: image_path, true_d10, true_d50, true_d90, etc.")
    parser.add_argument("--generated_test_csv", type=str, default="test.csv",
                        help="Where to save the newly created test CSV (image_path, d10, d50, d90).")
    
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to an existing test CSV with columns: image_path, d10, d50, d90.")
    
    parser.add_argument("--model_type", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "inception_v3"],
                        help="Model architecture to use.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the trained model weights (.pth).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference.")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="If set, do NOT use pretrained ImageNet weights.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu' or 'cuda'. If not set, auto-detect.")
    
    parser.add_argument("--output_csv", type=str, default="predictions.csv",
                        help="Path to output predictions CSV.")
    parser.add_argument("--plot_file", type=str, default="prediction_plot.png",
                        help="Where to save the scatter plot figure.")
    
    args = parser.parse_args()
    
    final_test_csv = None
    if args.old_pred_csv:
        generate_test_csv_from_predictions(args.old_pred_csv, args.generated_test_csv)
        final_test_csv = args.generated_test_csv
    elif args.test_csv:
        final_test_csv = args.test_csv
    else:
        raise ValueError("You must provide either --old_pred_csv or --test_csv.")
    
    if args.device is not None:
        device_str = args.device.lower()
        if device_str == "cpu":
            device = torch.device("cpu")
        elif device_str == "cuda":
            device = torch.device("cuda")
        else:
            raise ValueError("Unsupported device string. Use 'cpu' or 'cuda'.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.model_type == "inception_v3":
        input_size = 299
    else:
        input_size = 224
    
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_dataset = GranuleTestDataset(final_test_csv, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=1)
    
    num_images = len(test_dataset)
    
    model = get_model(args.model_type, pretrained=not args.no_pretrained)
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Weights file not found at {args.weights}")
        exit()

    all_preds, all_labels, all_paths = [], [], []
    inference_times = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            
            start_time = time.time()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            inference_times.append(time.time() - start_time)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    df_results = pd.DataFrame({
        "image_path": all_paths,
        "true_d10": all_labels[:, 0],
        "true_d50": all_labels[:, 1],
        "true_d90": all_labels[:, 2],
        "pred_d10": all_preds[:, 0],
        "pred_d50": all_preds[:, 1],
        "pred_d90": all_preds[:, 2],
    })
    df_results.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

    metrics_summary = {
        "r2_d10": r2_score_manual(all_labels[:, 0], all_preds[:, 0]),
        "r2_d50": r2_score_manual(all_labels[:, 1], all_preds[:, 1]),
        "r2_d90": r2_score_manual(all_labels[:, 2], all_preds[:, 2]),
        "mse_overall": mean_squared_error_manual(all_labels, all_preds),
        "r2_overall": r2_score_manual(all_labels, all_preds),
        "mae_overall": mean_absolute_error_manual(all_labels, all_preds),
        "avg_inference_time_ms": (sum(inference_times) / len(inference_times)) * 1000 if inference_times else 0,
    }

    print("\n--- Metrics Summary ---")
    for key, val in metrics_summary.items():
        print(f"{key.replace('_', ' ').title()}: {val:.4f}")
    print("-----------------------\n")

    plot_predictions(model, test_loader, device,
                     model_name=f"{args.model_type} (weights: {os.path.basename(args.weights)})",
                     save_path=args.plot_file)

if __name__ == '__main__':
    main()
