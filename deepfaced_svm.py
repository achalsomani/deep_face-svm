from sklearn.svm import SVC
import numpy as np
from tqdm import tqdm
import torch
from facenet_pytorch import InceptionResnetV1
import pickle
from dataset import TestDataset
from torch.utils.data import Dataset, DataLoader
from typing import List
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DeepFacedSVM:
    def __init__(self):
        self.feature_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.svm_classifier = SVC(kernel='rbf')

    def train(self, train_loader, num_epochs):
        train_features, train_labels = self._extract_features(train_loader, num_epochs)
        self.svm_classifier.fit(train_features, train_labels)
        train_accuracy = self.svm_classifier.score(train_features, train_labels)
        print(f"Train Accuracy: {train_accuracy}")

    def test(self, test_loader):
        test_features, test_labels = self._extract_features(test_loader, 1)
        test_accuracy = self.svm_classifier.score(test_features, test_labels)
        print(f"Test Accuracy: {test_accuracy}")

    def dump(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.svm_classifier, file)
        print(f"Model saved as {filename}")

    @classmethod
    def load_from_pickle(cls, filename):
        with open(filename, 'rb') as file:
            svm_classifier = pickle.load(file)
        instance = cls()
        instance.svm_classifier = svm_classifier
        return instance

    def _extract_features(self, loader, num_epochs):
        features = []
        labels = []

        for epoch in range(num_epochs):
            for images, batch_labels in tqdm(loader):
                images = images.to(device)

                with torch.no_grad():
                    embeddings = self.feature_extractor(images)

                features.append(embeddings.cpu().numpy())
                labels.extend(batch_labels)

        features = np.vstack(features)
        labels = np.vstack(labels)

        return features, labels


    def inference(self, image_list: List[np.ndarray]) -> np.ndarray:
        inference_dataset: TestDataset = TestDataset(image_list)
        inference_loader: DataLoader = DataLoader(inference_dataset, batch_size=1)

        test_features: np.ndarray
        _, test_features = self._extract_features(inference_loader, 1)
        labels: np.ndarray = self.svm_classifier.predict(test_features)

        return labels