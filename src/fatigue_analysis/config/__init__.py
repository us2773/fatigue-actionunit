"""YAML設定の読込と型付き設定モデル。"""

from fatigue_analysis.config.loader import init_config, load_config
from fatigue_analysis.config.models import AnalysisConfig

__all__ = ["AnalysisConfig", "init_config", "load_config"]
