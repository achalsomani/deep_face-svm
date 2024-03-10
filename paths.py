from pathlib import Path

PROJECT_PATH = Path(__file__).parent
DATA_PATH = PROJECT_PATH / "data"

TRAINING_DATA_PATH = DATA_PATH / "training"
TRAINING_CSV_FILE = TRAINING_DATA_PATH / "train_data_mapping.csv"

TESTING_DATA_PATH = DATA_PATH / "testing"
TESTING_CSV_FILE = TESTING_DATA_PATH / "test_data_mapping.csv"