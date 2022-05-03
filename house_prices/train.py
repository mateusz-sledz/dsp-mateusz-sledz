import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .preprocess import get_preprocessed_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import os
import joblib


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray,
                  precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def evaluate(model: RandomForestRegressor, x_test: np.ndarray,
             y_test: np.ndarray) -> float:
    y_pred = model.predict(x_test)
    return compute_rmsle(np.array(y_test), np.array(y_pred), 3)


def build_model(data: pd.DataFrame) -> dict[str, str]:
    y = data['SalePrice']
    x = data.drop(columns=['SalePrice'])

    train, test, y_train, y_test = train_test_split(x, y, random_state=0)

    train = get_preprocessed_data(train, is_train_data=True)
    test = get_preprocessed_data(test, is_train_data=False)

    rf = RandomForestRegressor(n_estimators=40, random_state=0)
    rf.fit(train, y_train)

    path = '../models/model.joblib'
    joblib.dump(rf, path)

    error = evaluate(rf, test, y_test)
    return {'rmsle': str(error), 'model_path': os.path.abspath(path)}
