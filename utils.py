from pathlib import Path
from torchvision.transforms import transforms

def save_images_with_labels(images, labels, folder_path, epoch):
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    for i in range(len(images)):
        image_name = f'epoch_{epoch}_label_{labels[i]}_image_{i}.png'
        image_path = folder_path / image_name
        transformed_image = transforms.ToPILImage()(images[i])
        transformed_image.save(image_path)
