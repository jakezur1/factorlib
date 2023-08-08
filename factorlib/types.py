from enum import Enum


class ModelType(Enum):
    xgboost = 'xgb'
    # catboost = 'catboost'
    lightgbm = 'lgbm'
    hist_gradient_boost = 'hgb'
    gradient_boost = 'gbr'
    adaboost = 'adb'
    random_forest = 'rf'


class SpliceBy(Enum):
    ticker = "ticker"
    date = "date"


class FactorlibUserWarning:
    TimeStep = 'TimeStepFormatting'
    ParameterOverride = 'ParameterOverride'
    NotImplemented = 'NotImplemented'
    AmbiguousInterval = 'AmbiguousInterval'
    MissingData = 'MissingData'


class PortOptOptions:
    MeanVariance = 'mvp'
    HierarchicalRiskParity = 'hrp'
    InverseVariance = 'iv'