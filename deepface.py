import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from deepface import DeepFace
from utils import get_data_loaders
from params import NUM_EPOCHS

def extract_deepface_features(images):
    # Initialize an empty list to store features
    features = []

    # Extract features for each image
    for image in images:
        img_representation = DeepFace.represent(image, model_name='DeepFace', enforce_detection=False)
        
        features.append(img_representation)

    return np.array(features)

class SVMFeatureExtractor:
    def __init__(self):
        self.svm_classifier = SVC(kernel='rbf')

    def train(self, features, labels):
        self.svm_classifier.fit(features, labels)

        train_accuracy = self.svm_classifier.score(features, labels)
        print(f"Train Accuracy: {train_accuracy}")

    def test(self, features, labels):
        # Evaluate the SVM classifier on the provided features and labels
        accuracy = self.svm_classifier.score(features, labels)
        print(f"Test Accuracy: {accuracy}")


def extract_features(train_loader, test_loader, num_epochs):
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    # Extract features for each epoch
    for epoch in range(num_epochs):
        train_features_epoch = []
        train_labels_epoch = []

        # Extract features for training data
        for images, labels_batch in train_loader:
            # Extract features using DeepFace
            batch_features = extract_deepface_features(images)
            train_features_epoch.extend(batch_features)
            train_labels_epoch.extend(labels_batch)

        # Convert lists to numpy arrays and store for this epoch
        train_features.append(np.array(train_features_epoch))
        train_labels.append(np.array(train_labels_epoch))

        # Extract features for testing data for just one epoch
        if epoch == 0:
            for images, labels_batch in test_loader:
                # Extract features using DeepFace
                batch_features = extract_deepface_features(images)
                test_features.extend(batch_features)
                test_labels.extend(labels_batch)

    # Convert test features and labels to numpy arrays
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    return train_features, train_labels, test_features, test_labels

# Example usage
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    train_features, train_labels, test_features, test_labels = extract_features(train_loader, test_loader, num_epochs=NUM_EPOCHS)
    train_features_concat = np.concatenate(train_features)
    train_labels_concat = np.concatenate(train_labels)
    train_svm_classifier(train_features_concat, train_labels_concat)
    test_svm_classifier(test_features, test_labels)  # You need to implement this function

def train_svm_classifier(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Initialize SVM classifier
    svm_classifier = SVC(kernel='rbf')

    # Train the SVM classifier
    svm_classifier.fit(X_train, y_train)

    # Evaluate the SVM classifier
    train_accuracy = svm_classifier.score(X_train, y_train)
    test_accuracy = svm_classifier.score(X_test, y_test)
    print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")