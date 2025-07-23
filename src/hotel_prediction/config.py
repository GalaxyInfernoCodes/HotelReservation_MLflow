# config.py
from typing import List, Optional
from pydantic import BaseModel, Field
import yaml
from pathlib import Path


# -------------------
# Config Models
# -------------------


class RandomForestConfig(BaseModel):
    n_estimators: int = Field(gt=0)
    max_depth: Optional[int] = Field(gt=0)
    min_samples_split: int = Field(gt=1)
    min_samples_leaf: int = Field(gt=1)
    random_state: int


class ModelConfigModel(BaseModel):
    random_forest: RandomForestConfig


class FeatureConfigModel(BaseModel):
    target_feature: List[str]
    categorical_features: List[str]
    numerical_features: List[str]


class ProjectConfigModel(BaseModel):
    data_source_path: str
    duckdb_data_path: str
    output_dir: str
    random_seed: int
    mlflow_tracking_uri: str
    # Proportions for dataset splitting
    val_size: float = Field(gt=0, lt=0.5, default=0.15)
    test_size: float = Field(gt=0, lt=0.5, default=0.15)


# -------------------
# Combined Wrapper Config
# -------------------


class FullConfig(BaseModel):
    project: ProjectConfigModel
    model: ModelConfigModel
    features: FeatureConfigModel


# -------------------
# Config Loader
# -------------------


def load_config(path: str | Path, model_cls: type[BaseModel]) -> BaseModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return model_cls(**data)


def load_full_config(
    project_path: str | Path = "project_config.yaml",
    model_path: str | Path = "model_config.yaml",
    feature_path: str | Path = "feature_config.yaml",
) -> FullConfig:
    return FullConfig(
        project=load_config(project_path, ProjectConfigModel),
        model=load_config(model_path, ModelConfigModel),
        features=load_config(feature_path, FeatureConfigModel),
    )
