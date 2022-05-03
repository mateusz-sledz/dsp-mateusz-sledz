import pandas as pd
import numpy as np
from .preprocess import get_preprocessed_data
import joblib


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    input_data = get_preprocessed_data(input_data, is_train_data=False)

    path = '../models/model.joblib'
    model = joblib.load(path)

    return model.predict(input_data)
