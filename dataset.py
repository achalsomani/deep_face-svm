import pandas as pd
from pathlib import Path
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from params import BATCH_SIZE, STARTING_SHAPE, INPUT_SHAPE

PROJECT_PATH = Path(__file__).parent
DATA_PATH = PROJECT_PATH / "data"

TRAINING_DATA_PATH = DATA_PATH / "training"
TRAINING_CSV_FILE = TRAINING_DATA_PATH / "train_data_mapping.csv"

TESTING_DATA_PATH = DATA_PATH / "testing"
TESTING_CSV_FILE = TESTING_DATA_PATH / "test_data_mapping.csv"

train_transforms = transforms.Compose([
    transforms.Resize(STARTING_SHAPE),
    transforms.RandomResizedCrop(INPUT_SHAPE, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=50),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SHAPE),
    transforms.ToTensor(),
])

class ClassDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.root_dir / self.data.iloc[idx, 0]
        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

class TestDataset(Dataset):
    def __init__(self, image_list, transform=test_transforms):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.fromarray(self.image_list[idx])
        if self.transform:
            image = self.transform(image)
        return image, None  # Labels are set as None

def get_data_loaders():
    train_dataset = ClassDataset(csv_file=TRAINING_CSV_FILE, root_dir=TRAINING_DATA_PATH, transform=train_transforms)
    test_dataset = ClassDataset(csv_file=TESTING_CSV_FILE, root_dir=TESTING_DATA_PATH, transform=test_transforms)
    
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, test_loader
