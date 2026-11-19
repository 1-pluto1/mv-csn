"""MV-CSN v2: Multi-View Context-Gated Character Similarity Network."""

from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.vocab import CharVocab
from mvcsn.inference import (
    DemoBundle,
    MVCSNInferenceEngine,
    PredictionResult,
    load_demo_bundle,
    save_demo_bundle,
)
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.utils.config import MVCSNConfig, TrainConfig

__version__ = "2.0.0"

__all__ = [
    "CharVocab",
    "DemoBundle",
    "MVCSNConfig",
    "MVCSNInferenceEngine",
    "MVCSNModel",
    "PredictionResult",
    "TextPreprocessor",
    "TrainConfig",
    "load_demo_bundle",
    "save_demo_bundle",
    "__version__",
]
