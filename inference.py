from deepfaced_svm import DeepFacedSVM

deepfaced_svm_model = DeepFacedSVM.load_from_pickle("deepfaced_svm_model.pkl")

deepfaced_svm_model.inference()