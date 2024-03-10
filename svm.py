from sklearn.svm import SVC

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
