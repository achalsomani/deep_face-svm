from dataset import get_data_loaders
from params import NUM_EPOCHS
from deepfaced_svm import DeepFacedSVM

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    
    deepfaced_svm = DeepFacedSVM()
    deepfaced_svm.train(train_loader, NUM_EPOCHS)
    deepfaced_svm.test(test_loader)
    deepfaced_svm.dump("deepfaced_svm_model.pkl")
