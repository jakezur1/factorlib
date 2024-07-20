# FactorLib

A python library for creating custom factors, and running walk-forward optimization with various portfolio optimization options.

### Features

- Parallel processing for custom factor data creation with highly customizable interface.
- Walk forward optimization with various portfolio optimization options. 
- Performance visualization and sharpe ratios for benchmarking. 
- Integration with SHAP values for model interpretability.

### Prerequisites

The following software needs to be installed to use FactorLib:
* python 
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

To install all of the required dependencies run:
```sh
pip install -r requirements.txt
```

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Install pip packages
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### factorlib

The factorlib directory contains all of the source code for the FactorLib library. The most important files to get
started are:
    
    - **factor.py**: Defines the `Factor` class, which is used to store and prepare data to later be passed into
     a `FactorModel`.
    - **factor\_model.py**: Defines the `FactorModel` class, which is responsible for training and evaluating a model
    using the provided factors. 

The recommended workflow for using FactorLib is as follows:
1. Prepare your factor data in a pandas DataFrame. The DataFrame should have a multi-index with levels named 'date'
 and 'ticker'. Each column in the DataFrame represents a different factor. 
2. Instantiate a `Factor` object, passing in the factor data and other relevant parameters, such as the factor name, the time 
interval for the data (e.g. daily, monthly), and any transformations to apply to the data. 
3. Instantiate a `FactorModel` object, passing in the desired model type (e.g. LightGBM), tickers used for training
 and evaluation, and the time interval for the data.
4. Add the `Factor` object to the `FactorModel` using the `add_factor()` method. 
5. Call the `wfo()` (walk-forward optimization) method on the `FactorModel` object. This will train and evaluate
the model over the specified time period, using a rolling window approach. The `wfo()` method returns a `Statistics`
 object that contains various performance metrics for the model. 
6. Analyze the results using the various methods provided by the `Statistics` object, such as `snapshot()`, `stats_report`,
`beeswarm_shaps()` and `waterfall_shaps()`.

The file **debug.py** is an example of how to use the `Factor` and `FactorModel` classes in conjunction to train a 
model and visualize the results.


