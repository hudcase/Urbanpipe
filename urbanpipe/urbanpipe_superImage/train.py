import os
import torch
import torch.optim as optim
import torch.nn as nn
from helper.dataset import get_dataset_urbanpipe
from helper.models import get_model
from helper.utils import train, test
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def main(args):

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Get data loaders
    train_loader, test_loader = get_dataset_urbanpipe(args)

    # Get the model and move it to GPU
    model = get_model(model_name=args.model_name, num_classes=args.num_classes)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # Training and validation process
    train_losses = []
    train_mAPs = []  # Mean Average Precision (mAP) on training set
    test_losses = []
    test_mAPs = []   # Mean Average Precision (mAP) on test set
    best_mAP = 0
    for epoch in tqdm(range(args.epochs)):
        # Train
        train_loss, train_mAP = train(model, train_loader, criterion, optimizer, epoch, device)
        train_losses.append(train_loss)
        train_mAPs.append(train_mAP)
        
        # Test
        test_loss, test_mAP = test(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_mAPs.append(test_mAP)

        # log best mAP
        if test_mAP > best_mAP:
            best_mAP = test_mAP

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.3f}, Train mAP: {train_mAP:.3f}, Test Loss: {test_loss:.3f}, Test mAP: {test_mAP:.3f}, Best mAP: {best_mAP:.3f} \n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Urban Pipe Training Script")
    parser.add_argument("--base_path", type=str, default="./dataset/superImage", help="Base path of the dataset")
    parser.add_argument("--label_file_train", type=str, default="../dataset/video_label_file/video_train.json", help="Path to the training label file")
    parser.add_argument("--label_file_val", type=str, default="../dataset/video_label_file/video_val.json", help="Path to the validation label file")
    parser.add_argument("--model_name", type=str, default="vit_h_14", help="Model name, default is a custom model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and testing")
    parser.add_argument("--num_classes", type=int, default=17, help="Number of classes in the dataset")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the SGD optimizer")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs for training")
    args = parser.parse_args()

    main(args)
