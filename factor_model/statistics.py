import pandas as pd
import numpy as np
import quantstats as qs
from factor_model.factor_model import FactorModel


class Statistics:
    def __init__(self, portfolio_returns, model: FactorModel):
        self.portfolio_returns = portfolio_returns
        self.model = model

    def get_full_qs(self):
        qs.reports.full(self.portfolio_returns)

    def find_factor_significance(self):
        pass
