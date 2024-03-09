import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from torchvision.transforms import transforms
from paths import DATA_PATH, CSV_FILE, TESTING_DATA_PATH, TESTING_CSV_FILE
from params import BATCH_SIZE, device

def get_data_loaders(train_transforms, test_transforms):
    # Load train and test datasets
    train_dataset = CustomDataset(csv_file=CSV_FILE, root_dir=DATA_PATH, transform=train_transforms)
    test_dataset = CustomDataset(csv_file=TESTING_CSV_FILE, root_dir=TESTING_DATA_PATH, transform=test_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def save_images_with_labels(images, labels, folder_path, epoch):
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    for i in range(len(images)):
        image_name = f'epoch_{epoch}_label_{labels[i]}_image_{i}.png'
        image_path = folder_path / image_name
        transformed_image = transforms.ToPILImage()(images[i])
        transformed_image.save(image_path)
