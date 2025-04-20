import logging
import pandas as pd
from src.hotel_prediction.config import FullConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, config: FullConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        hotel_dataframe = pd.read_csv(self.config.project.data_path)
