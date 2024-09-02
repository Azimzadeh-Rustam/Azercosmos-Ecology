import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats import pearsonr
import numpy as np


def filter_outliers(x, y):
    data = np.column_stack((x, y))
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    is_anomaly = (data < lower_bound) | (data > upper_bound)
    is_anomaly = is_anomaly.any(axis=1)

    anomalies = data[is_anomaly]
    normal_data = data[~is_anomaly]

    return normal_data, anomalies


def main():
    forests_data = pd.read_json('../03_results/04_correlation_matrix/forests.json', convert_dates=['dates'])
    sea_data = pd.read_json('../03_results/04_correlation_matrix/sea.json', convert_dates=['dates'])
    data = pd.merge(forests_data, sea_data, on='dates')
    data.set_index('dates', inplace=True)

    NDVI = data['NDVI'].values
    WCI = data['WCI'].values

    filtered_data, outliers = filter_outliers(NDVI, WCI)
    dependent_filtered_data = filtered_data[:, 0]
    independent_filtered_data = filtered_data[:, 1]
    dependent_outliers = outliers[:, 0]
    independent_outliers = outliers[:, 1]

    corr, p_value = pearsonr(dependent_filtered_data, independent_filtered_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(dependent_filtered_data, independent_filtered_data, label='Нормальные данные', color='blue')
    plt.scatter(dependent_outliers, independent_outliers, label='Аномалии', color='red')
    plt.xlabel('NDVI')
    plt.ylabel('WCI')
    plt.title('Корреляция между NDVI и WCI с выделением аномалий')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
