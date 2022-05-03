from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
import pandas as pd


def transform_to_time_distance(data: pd.DataFrame) -> pd.DataFrame:
    data['YearBuilt'] = 2022 - data['YearBuilt']
    data['YearRemodAdd'] = 2022 - data['YearRemodAdd']
    data['YrSold'] = 2022 - data['YrSold']

    return data


def encode_categorical(data: pd.DataFrame,
                       is_train_data: bool) -> pd.DataFrame:
    categorical_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley',
                        'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                        'LandSlope', 'Neighborhood', 'Condition1',
                        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                        'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual',
                        'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                        'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                        'KitchenQual', 'Functional', 'FireplaceQu',
                        'GarageType', 'GarageFinish', 'GarageQual',
                        'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                        'MiscFeature', 'SaleCondition']

    data = data.fillna(value='nan_but_category')

    if is_train_data:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                 unknown_value=40)
        encoder = encoder.fit(data[categorical_cols])
        joblib.dump(encoder, '../models/encoder.joblib')
    else:
        encoder = joblib.load('../models/encoder.joblib')

    data[categorical_cols] = encoder.transform(data[categorical_cols])

    return data


def fill_missing_continuous(data: pd.DataFrame, columns: list[str],
                            is_train_data: bool) -> pd.DataFrame:
    if is_train_data:
        medians = data[columns].median()
        joblib.dump(medians, '../models/medians.joblib')
    else:
        medians = joblib.load('../models/medians.joblib')

    if not data[columns].isna().any().any():
        return data

    data[columns] = data[columns].fillna(value=medians)

    return data


def scale_continuous(data: pd.DataFrame, is_train_data: bool) -> pd.DataFrame:
    continuous_cols = ['LotArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                       'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                       'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                       'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                       'MiscVal']

    data = fill_missing_continuous(data, continuous_cols, is_train_data)

    if is_train_data:
        scaler = StandardScaler()
        scaler = scaler.fit(data[continuous_cols])
        joblib.dump(scaler, '../models/scaler.joblib')
    else:
        scaler = joblib.load('../models/scaler.joblib')

    data[continuous_cols] = scaler.transform(data[continuous_cols])
    return data


def get_preprocessed_data(data: pd.DataFrame,
                          is_train_data: bool) -> pd.DataFrame:
    to_drop = ['Id', 'MasVnrArea', 'MasVnrType', 'BsmtFullBath',
               'BsmtHalfBath', 'GarageYrBlt', 'MoSold',
               'SaleType', 'LotFrontage']

    x = data.drop(columns=to_drop)
    x = transform_to_time_distance(x)
    x = scale_continuous(x, is_train_data)
    x = encode_categorical(x, is_train_data)

    return x
