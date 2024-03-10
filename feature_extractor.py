import numpy as np
from tqdm import tqdm
import torch

from facenet_pytorch import InceptionResnetV1


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load VGGFace model
vgg_face = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_deepface_features_from_loader(loader):
    features = []
    labels = []
    for images, batch_labels in tqdm(loader):
        # Convert images to tensor and move to GPU if available
        images = images.to(device)

        # Extract features
        with torch.no_grad():
            embeddings = vgg_face(images)

        # Append features and labels
        features.append(embeddings.cpu().numpy())
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