"""利用者へ説明可能な例外型。"""


class FatigueAnalysisError(Exception):
    """本ツールで利用者向けに扱う基底例外。"""


class ConfigError(FatigueAnalysisError):
    """YAML設定の構造、型、値が契約に合わない場合の例外。"""


class InputContractError(FatigueAnalysisError):
    """入力CSVや実データが研究契約に合わない場合の例外。"""


class OutputConflictError(FatigueAnalysisError):
    """既存成果物を暗黙に上書きしそうな場合の例外。"""


class NodeExecutionError(FatigueAnalysisError):
    """node内の数値処理や成果物生成が失敗した場合の例外。"""
