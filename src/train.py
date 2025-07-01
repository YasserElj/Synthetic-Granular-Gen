import os
import argparse
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
import random
import torchvision.transforms.functional as F

# -------------------------------
# Custom Dataset Definitions
# -------------------------------
class GranuleDataset(Dataset):
    def __init__(self, csv_file, transform=None, return_path=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.return_path = return_path
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_transformed = self.transform(image)
        else:
            image_transformed = image
        target = np.array([row['d10'], row['d50'], row['d90']], dtype=np.float32)
        if self.return_path:
            return image_transformed, torch.tensor(target), img_path
        else:
            return image_transformed, torch.tensor(target)

class GranuleDatasetFromDF(Dataset):
    def __init__(self, df, transform=None, return_path=False):
        self.data = df.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_transformed = self.transform(image)
        else:
            image_transformed = image
        target = np.array([row['d10'], row['d50'], row['d90']], dtype=np.float32)
        if self.return_path:
            return image_transformed, torch.tensor(target), img_path
        else:
            return image_transformed, torch.tensor(target)

# -------------------------------
# Optional: Custom Transform for Fixed Rotations
# -------------------------------
class RandomFixedRotation:
    def __init__(self, angles):
        self.angles = angles
    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)

# -------------------------------
# Model Selection Function
# -------------------------------
def get_model(model_type):
    if model_type == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)  # Train from scratch
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 3)
    elif model_type == "cnn":
        # A simple custom CNN for regression.
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2)
                )
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 3)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        model = SimpleCNN()
    elif model_type == "efficientnet_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 3)
    elif model_type == "inception_v3":
        from torchvision.models import inception_v3, Inception_V3_Weights
        # Pretrained inception_v3 requires aux_logits=True.
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 3)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model

# -------------------------------
# Results Folder Creation
# -------------------------------
def create_results_folder(model_type, base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    idx = 1
    while True:
        folder_name = f"{model_type}_{idx:02d}"
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path
        idx += 1

# -------------------------------
# Visualization Functions
# -------------------------------
def plot_predictions(model, test_loader, device, save_path):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    metrics = ['d10', 'd50', 'd90']
    plt.figure(figsize=(18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(all_labels[:, i], all_predictions[:, i], alpha=0.7, label=f'Predicted vs True {metrics[i]}')
        plt.plot([all_labels[:, i].min(), all_labels[:, i].max()],
                 [all_labels[:, i].min(), all_labels[:, i].max()], 'r--')
        plt.xlabel(f'True {metrics[i]}')
        plt.ylabel(f'Predicted {metrics[i]}')
        plt.title(f'{metrics[i]}: Prediction vs Ground Truth')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction plot saved to {save_path}")
    return all_labels, all_predictions

def visualize_test_predictions(model, test_loader, device, results_folder, num_images=5):
    model.eval()
    images_shown = 0
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
                paths = [None] * images.size(0)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            for i in range(images.size(0)):
                if images_shown >= num_images:
                    return
                if paths[i] is not None:
                    original_img = Image.open(paths[i]).convert('RGB')
                    original_img_np = np.array(original_img) / 255.0
                else:
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    original_img_np = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
                    original_img_np = np.clip(original_img_np, 0, 1)
                pred = outputs[i].cpu().numpy()
                true = labels[i].cpu().numpy()
                plt.figure(figsize=(6, 6))
                plt.imshow(original_img_np)
                plt.axis('off')
                plt.title(f"Predicted: d10={pred[0]:.2f}, d50={pred[1]:.2f}, d90={pred[2]:.2f}\n"
                          f"True: d10={true[0]:.2f}, d50={true[1]:.2f}, d90={true[2]:.2f}")
                img_save_path = os.path.join(results_folder, f"test_prediction_{images_shown}.png")
                plt.savefig(img_save_path)
                plt.close()
                print(f"Test prediction image saved to {img_save_path}")
                images_shown += 1

def plot_training_loss(train_losses, val_losses, save_path):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training loss plot saved to {save_path}")

def compute_test_metrics(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    criterion = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    mse_d10 = criterion(torch.tensor(all_predictions[:, 0]), torch.tensor(all_labels[:, 0])).item()
    mse_d50 = criterion(torch.tensor(all_predictions[:, 1]), torch.tensor(all_labels[:, 1])).item()
    mse_d90 = criterion(torch.tensor(all_predictions[:, 2]), torch.tensor(all_labels[:, 2])).item()
    overall_mse = criterion(torch.tensor(all_predictions), torch.tensor(all_labels)).item()
    metrics = {
        "mse_d10": mse_d10,
        "mse_d50": mse_d50,
        "mse_d90": mse_d90,
        "overall_mse": overall_mse
    }
    return metrics

def save_predictions_csv(model, test_loader, device, csv_save_path):
    model.eval()
    predictions = []
    ground_truths = []
    image_paths = []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
                paths = [None] * images.size(0)
            images = images.to(device)
            outputs = model(images)
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())
            image_paths.extend(paths)
    df = pd.DataFrame({
        "image_path": image_paths,
        "true_d10": [gt[0] for gt in ground_truths],
        "true_d50": [gt[1] for gt in ground_truths],
        "true_d90": [gt[2] for gt in ground_truths],
        "pred_d10": [pred[0] for pred in predictions],
        "pred_d50": [pred[1] for pred in predictions],
        "pred_d90": [pred[2] for pred in predictions],
    })
    df.to_csv(csv_save_path, index=False)
    print(f"Predictions CSV saved to {csv_save_path}")

# -------------------------------
# Main Training Function
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train model on granule dataset with TensorBoard logging")
    parser.add_argument("--csv", type=str, default="data/granules_dataset.csv", help="Path to dataset CSV file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_type", type=str, default="resnet50", 
                        choices=["resnet50", "cnn", "efficientnet_b0", "inception_v3"],
                        help="Model architecture to use")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Set input resolution based on model type.
    if args.model_type == "inception_v3":
        input_size = 299
    else:
        input_size = 224

    # Define transforms.
    # For training, we add augmentation (rotation and color jitter). Here you could use either a continuous random rotation or a custom one.
    train_transform = transforms.Compose([
        # For discrete rotations, you can use RandomFixedRotation:
        RandomFixedRotation([0, 90, 180, 270]),
        # transforms.RandomRotation(degrees=90),  # This rotates between -90 and +90 degrees randomly.
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Read CSV into DataFrame and split into train/validation.
    df = pd.read_csv(args.csv)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_dataset = GranuleDatasetFromDF(train_df, transform=train_transform, return_path=True)
    val_dataset = GranuleDatasetFromDF(val_df, transform=val_transform, return_path=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    # Get model.
    model = get_model(args.model_type).to(device)
    
    # Loss, optimizer, and TensorBoard writer.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = os.path.join("runs", args.model_type, run_id)
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    # Create unique results folder.
    results_folder = create_results_folder(args.model_type)
    
    train_losses = []
    val_losses = []
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):  # If aux_logits are enabled, outputs is (main_output, aux_output)
                outputs = outputs[0]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            global_step += 1
            if global_step % 10 == 0:
                writer.add_scalar("Loss/Train", loss.item(), global_step)
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar("Epoch/Train_Loss", avg_train_loss, epoch)
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.4f}")
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        writer.add_scalar("Epoch/Val_Loss", avg_val_loss, epoch)
        print(f"Epoch [{epoch+1}/{args.epochs}] Val Loss: {avg_val_loss:.4f}")
        
        # Save best model if current validation loss is lower.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(results_folder, f"{args.model_type}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with val loss {best_val_loss:.4f}")
    
    # Save final model.
    final_model_path = os.path.join(results_folder, f"{args.model_type}_granule.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    writer.close()
    
    # Save training loss plot.
    loss_plot_path = os.path.join(results_folder, "training_loss.png")
    plot_training_loss(train_losses, val_losses, loss_plot_path)
    
    # Compute test metrics.
    test_metrics = compute_test_metrics(model, val_loader, device)
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
    metrics_file = os.path.join(results_folder, "test_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Test Metrics:\n")
        f.write(metrics_text)
    print(f"Test metrics saved to {metrics_file}")
    
    # Save prediction plot.
    pred_plot_path = os.path.join(results_folder, "predictions_vs_truth.png")
    plot_predictions(model, val_loader, device, pred_plot_path)
    
    # Save CSV of predictions vs. ground truth.
    csv_save_path = os.path.join(results_folder, "predictions.csv")
    save_predictions_csv(model, val_loader, device, csv_save_path)
    
    # Visualize some test predictions.
    visualize_test_predictions(model, val_loader, device, results_folder, num_images=5)

if __name__ == "__main__":
    main()
