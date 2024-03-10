from feature_extractor import extract_features
import numpy as np

# Load your test data here
# Assuming you have test data stored in a file named "test_data.npy"
test_data = np.load("test_data.npy")
features_test = test_data[:, :-1]  # Assuming the last column is labels
labels_test = test_data[:, -1]  # Assuming labels are in the last column

# Load the trained SVM model
extractor = extract_features.load_from_pickle("svm_model.pkl")

# Test the model
extractor.test(features_test, labels_test)

# Now, you can use the trained model to make predictions on new data
# For example:
new_features = np.array([[...], [...], ...])  # New feature data
predictions = extractor.svm_classifier.predict(new_features)
print("Predictions:", predictions)
