import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 10
STARTING_SHAPE = (250, 250)
INPUT_SHAPE = (224, 224)

LEARNING_RATE = 0.001
NUM_EPOCHS = 3