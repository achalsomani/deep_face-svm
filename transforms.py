from torchvision.transforms import transforms
from params import STARTING_SHAPE, INPUT_SHAPE

train_transforms = transforms.Compose([
    transforms.Resize(STARTING_SHAPE),  # Resize to 256x256
    transforms.RandomResizedCrop(INPUT_SHAPE, scale=(0.8, 1.0), ratio=(0.8, 1.2)),  # Random crop to 224x224
    transforms.ColorJitter(brightness=0.5),  # Randomly adjust brightness
    transforms.RandomRotation(degrees=50),  # Random rotation
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.GaussianBlur(kernel_size=3),  # Apply Gaussian blur
    
    transforms.ToTensor(),
])

# Define modified test transforms
test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SHAPE),  # Resize to INPUT_SHAPE
    transforms.ToTensor(),
])