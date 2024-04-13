# FactorLib: A Library for Financial Factor Modeling and Walk-Forward Optimization

This project provides a comprehensive Python library called FactorLib for financial factor modeling and walk-forward optimization (WFO). It allows users to create custom factors, build factor models, and perform WFO to evaluate and optimize portfolio strategies. 

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


## Features

- **Custom Factor Creation:** Easily create custom factors with parallel processing using the BaseFactor class.
- **Factor Model Construction:** Build factor models using the Factor and FactorModel classes, incorporating various factor data sources.
- **Walk-Forward Optimization:** Conduct WFO to evaluate and optimize factor models across different time periods, with various portfolio construction options.
- **Performance Analysis:** Generate comprehensive performance statistics and visualizations, including information coefficient (IC), Sharpe ratio, and portfolio snapshots.
- **Flexibility:** Supports various machine learning models for factor modeling, such as LightGBM, XGBoost, and more.


## Prerequisites

To use FactorLib, ensure you have the following dependencies installed:

* pandas
* numpy
* scikit-learn
* scipy
* xgboost
* ray
* tqdm
* jupyter
* shap
* catboost
* lightgbm
* QuantStats
* matplotlib
* pyarrow
* fastparquet
* ipywidgets
* yfinance
* prettytable

**Installation:**

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn scipy xgboost ray tqdm jupyter shap catboost lightgbm QuantStats matplotlib pyarrow fastparquet ipywidgets yfinance prettytable
```



## Getting Started

### Usage

1. **Create Custom Factors (Optional):**
   - Inherit from the `BaseFactor` class and implement the `generate_data` method to define your factor logic.
   - Refer to the K-means clustering example in `factorlib/base_factor.py` for guidance.
2. **Prepare Factor Data:**
   - Ensure your factor data is formatted as a pandas DataFrame with appropriate index levels (e.g., 'date' and 'ticker').
   - Refer to the `Factor` class documentation in `factorlib/factor.py` for data formatting details.
3. **Build Factor Model:**
   - Create a `FactorModel` instance, specifying the model name, tickers, and interval.
   - Use the `add_factor` method to incorporate your factors into the model. 
4. **Perform Walk-Forward Optimization:**
   - Call the `wfo` method on your `FactorModel` instance, providing returns data, training parameters, and WFO settings.
5. **Analyze Results:**
   - Access performance statistics and visualizations using the returned `Statistics` object.

### Example Workflow

```python
import pandas as pd
from factorlib.factor import Factor
from factorlib.factor_model import FactorModel
from factorlib.types import ModelType

# Load factor and returns data (replace with your actual data sources)
factors = pd.read_csv('path/to/factors.csv')
returns = pd.read_csv('path/to/returns.csv')

# Create Factor objects
factor1 = Factor(name='factor1', data=factors[['factor1_column']])
factor2 = Factor(name='factor2', data=factors[['factor2_column']])

# Build FactorModel
model = FactorModel(name='my_model', tickers=returns['ticker'].unique(), interval='D', model_type=ModelType.lightgbm)
model.add_factor(factor1)
model.add_factor(factor2)

# Perform WFO (adjust parameters as needed)
stats = model.wfo(returns, train_interval=pd.DateOffset(years=5), start_date=pd.to_datetime('2018-01-01'),
                  end_date=pd.to_datetime('2023-01-01'))

# Analyze results
stats.stats_report()
stats.snapshot()
```


## Deployment

The provided code base focuses on research and development. Deployment would involve integrating the factor modeling and WFO pipeline into a production environment, potentially using tools like Docker and cloud platforms.


## License

This project is licensed under the MIT License. 


## Acknowledgments

- Choose an Open Source License
- GitHub Emoji Cheat Sheet
- Malven's Flexbox Cheatsheet
- Malven's Grid Cheatsheet
- Img Shields
- GitHub Pages
- Font Awesome
- React Icons

