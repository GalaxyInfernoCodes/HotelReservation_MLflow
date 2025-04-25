import logging

from hotel_prediction.data_loader import DataLoader
from hotel_prediction.config import load_full_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s]: \n%(message)s"
)

full_config = load_full_config(
    project_path="../config/project_config.yaml",
    model_path="../config/model_config.yaml",
    feature_path="../config/feature_config.yaml",
)


data_loader = DataLoader(config=full_config)
X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_dataset(
    data_loader.hotel_dataframe
)
data_loader.save_splits(X_train, X_val, X_test, y_train, y_val, y_test)