# lane_detection.py

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image


# CNN Model for Lane Detection

class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        return x

# Custom Dataset for Lane Detection
class LaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = glob(os.path.join(image_dir, '*.jpg'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Load Dataset

def load_dataset(image_dir, mask_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = LaneDataset(image_dir, mask_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


# Train Function

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, masks in tqdm(train_loader):
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# Evaluate Function

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

    return running_loss / len(val_loader)


# Save Model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


# Real-time Lane Detection

def detect_lanes(model, input_source, output_path=None):
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        return

    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor).cpu().squeeze(0).squeeze(0).numpy()
            output = (output > 0.5).astype(np.uint8) * 255

        # Overlay lane on the frame
        lane_overlay = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        combined_frame = cv2.addWeighted(frame, 0.8, lane_overlay, 0.2, 0)

        # Show frame
        cv2.imshow("Lane Detection", combined_frame)

        if output_path:
            out.write(combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


# Main Function

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Lane Detection using PyTorch and OpenCV")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'detect'])
    parser.add_argument('--train_images', type=str, help="Path to training images")
    parser.add_argument('--train_masks', type=str, help="Path to training masks")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--video_source', type=str, help="Path to video or 'webcam'")
    parser.add_argument('--output', type=str, help="Path to save output video")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LaneNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.mode == 'train':
        train_loader = load_dataset(args.train_images, args.train_masks, args.batch_size)
        for epoch in range(args.epochs):
            loss = train(model, train_loader, criterion, optimizer, device)
            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {loss:.4f}")
            save_model(model, 'lane_detection.pth')

    elif args.mode == 'detect':
        model.load_state_dict(torch.load('lane_detection.pth'))
        detect_lanes(model, args.video_source, args.output)

if __name__ == "__main__":
    main()
