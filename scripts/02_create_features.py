import logging

from hotel_prediction.config import load_full_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s]: \n%(message)s"
)

full_config = load_full_config(
    project_path="./config/project_config.yaml",
    model_path="./config/model_config.yaml",
    feature_path="./config/feature_config.yaml",
)
