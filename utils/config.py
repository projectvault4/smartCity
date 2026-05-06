from dataclasses import dataclass, field
from pathlib import Path

import torch


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    data_dir: Path = Path("data")
    data_file: Path = Path("data/urban_multivariate_timeseries.csv")
    dataset_dir: Path = Path("datasets")
    raw_traffic_file: Path = Path("Banglore_traffic_Dataset_raw.csv")
    raw_aqi_file: Path = Path("AirQualityUCI.csv")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    plot_dir: Path = Path("outputs/plots")
    random_seed: int = 42

    num_steps: int = 1600
    seq_len: int = 8
    forecast_horizon: int = 1
    direct_forecast_steps: int = 24
    lag_steps: tuple = (1, 2, 3, 6, 12, 24)
    rolling_windows: tuple = (6, 12, 24)
    closeness_lags: tuple = tuple(range(1, 9))
    period_lags: tuple = tuple(range(24, 32))
    trend_lags: tuple = tuple(range(24 * 7, 24 * 7 + 8))
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    target_col: str = "aqi"
    target_columns: tuple = field(
        default_factory=lambda: ("traffic_flow", "aqi", "temperature", "electricity_demand")
    )
    scaler_name: str = "minmax"

    bilstm_hidden_dim: int = 48
    bilstm_layers: int = 2
    tft_hidden_dim: int = 64
    tft_heads: int = 4
    tft_layers: int = 2
    tft_ff_dim: int = 128
    transformer_d_model: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff_dim: int = 128
    informer_d_model: int = 64
    informer_heads: int = 4
    informer_layers: int = 2
    informer_ff_dim: int = 128
    patchtst_d_model: int = 64
    patchtst_heads: int = 4
    patchtst_layers: int = 2
    patchtst_ff_dim: int = 128
    patchtst_patch_len: int = 4
    patchtst_stride: int = 2
    dense_hidden_dim: int = 64
    dropout: float = 0.2

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    patience: int = 3
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5

    streaming_window: int = 200
    streaming_step: int = 24
    drift_error_window: int = 48
    drift_threshold: float = 1.2
    ensemble_error_window: int = 72
    device: str = field(default_factory=_default_device)

    domain_columns: tuple = field(
        default_factory=lambda: ("traffic_flow", "aqi", "electricity_demand", "temperature", "humidity")
    )
    granger_max_lag: int = 6
    adaptive_switch_window: int = 48
    uncertainty_quantiles: tuple = (0.1, 0.9)


CONFIG = Config()
