import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = len(self.data['label'].unique())

    def __len__(self):
        return len(self.data)
        #return 4
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]
        transformed_image = self.transform(image)
        return transformed_image, label
    
    def numeric_to_string_label(self, numeric_label):
        return self.label_mapping[numeric_label]

