from utils import get_data_loaders
from params import NUM_EPOCHS
from svm import SVMFeatureExtractor
from feature_extractor import extract_features
from transforms import train_transforms, test_transforms


if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(train_transforms, test_transforms)
    x_train_features, y_train_labels, x_test_features, y_test_labels = \
        extract_features(train_loader,test_loader,NUM_EPOCHS)
    svm_classifier = SVMFeatureExtractor()
    svm_classifier.train(x_train_features, y_train_labels)
    svm_classifier.test(x_test_features, y_test_labels)



