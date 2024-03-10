from sklearn.svm import SVC
import pickle

class SvmClassifier:
    def __init__(self):
        self.svm_classifier = SVC(kernel='rbf')

    def train(self, features, labels):
        self.svm_classifier.fit(features, labels)
        pred = self.svm_classifier.predict(features)

        train_accuracy = self.svm_classifier.score(features, labels)
        print(f"Train Accuracy: {train_accuracy}")

    def test(self, features, labels):
        # Evaluate the SVM classifier on the provided features and labels
        accuracy = self.svm_classifier.score(features, labels)
        print(f"Test Accuracy: {accuracy}")

    def dump(self, filename):
        # Dump the trained SVM classifier to a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(self.svm_classifier, file)

    @classmethod
    def load_from_pickle(cls, filename):
        # Load the trained SVM classifier from a pickle file
        with open(filename, 'rb') as file:
            svm_classifier = pickle.load(file)
        instance = cls()
        instance.svm_classifier = svm_classifier
        return instance