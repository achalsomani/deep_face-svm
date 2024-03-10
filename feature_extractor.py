import numpy as np
from sklearn.model_selection import train_test_split
from deepface import DeepFace
from utils import get_data_loaders
from params import NUM_EPOCHS
from tqdm import tqdm

def extract_deepface_features_from_loader(loader):
    features = []
    labels = []

    for images, batch_labels in tqdm(loader):
        images_numpy = images.permute(0, 2, 3, 1).numpy()
        
        for image_numpy in images_numpy:
            image_features = DeepFace.represent(image_numpy, model_name='DeepFace', enforce_detection=False)
            features.append(image_features[0]["embedding"]) 
        labels.extend(batch_labels)

    features = np.vstack(features)
    labels = np.vstack(labels)

    return features, labels

def extract_features(train_loader, test_loader, num_epochs):
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    # Extract features for each epoch
    for epoch in range(num_epochs):
        # Extract features and labels for training data
        train_features_epoch, train_labels_epoch = extract_deepface_features_from_loader(train_loader)
        train_features.append(train_features_epoch)
        train_labels.append(train_labels_epoch)
    
    # Concatenate features and labels
    train_features = np.vstack(train_features)
    train_labels = np.concatenate(train_labels)

    # Extract features and labels for testing data for just one epoch
    test_features, test_labels = extract_deepface_features_from_loader(test_loader)

    return train_features, train_labels, test_features, test_labels