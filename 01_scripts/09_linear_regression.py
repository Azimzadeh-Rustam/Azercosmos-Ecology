import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np


def detect_outliers(data, independent, dependent, eps=0.5, min_samples=2):
    considered_features = data[[independent, dependent]]
    features = considered_features.values

    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(features)

    labels = model.labels_
    is_outlier = labels == -1

    filtered_data = data.loc[~is_outlier]
    outliers = data.loc[is_outlier]

    return filtered_data, outliers


def build_linear_regression(data, independent, dependent):
    X = sm.add_constant(data[independent])
    y = data[dependent]
    model = sm.OLS(y, X).fit()
    return model, X, y


def plot_regression(X, y, model, independent, dependent, outliers):
    plt.figure(figsize=(12, 8))
    plt.scatter(X[independent], y, color='blue', label='Actual Data')
    plt.plot(X[independent], model.predict(X), color='red', label='Fitted Line')

    prstd, iv_l, iv_u = wls_prediction_std(model)
    plt.plot(X[independent], iv_l, 'r--', label='Lower Confidence Limit (95%)')
    plt.plot(X[independent], iv_u, 'r--', label='Upper Confidence Limit (95%)')

    if not outliers.empty:
        plt.scatter(outliers[independent], outliers[dependent], color='red', s=50, edgecolor='black', alpha=0.65,
                    label='Anomalies')

    pearson_r = np.corrcoef(X[independent], y)[0, 1]

    coef = model.params.iloc[1]
    intercept = model.params.iloc[0]
    p_value = model.pvalues.iloc[1]
    std_error = model.bse.iloc[1]

    stats_text = (f'Coefficient (β): {coef:.4f}\n'
                  f'Intercept (α): {intercept:.4f}\n'
                  f'R-squared: {model.rsquared:.4f}\n'
                  f'P-value: {p_value:.4g}\n'
                  f'Standard Error: {std_error:.4f}\n'
                  f'Pearson r: {pearson_r:.4f}')
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel(independent)
    plt.ylabel(dependent)
    plt.title(f'Linear Regression Analysis\n{dependent} vs {independent}')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()


def main():
    forests_data = pd.read_json('../02_results/04_forests_indices/forests.json', convert_dates=['dates'])
    sea_data = pd.read_json('../02_results/05_sea_indices/sea.json', convert_dates=['dates'])
    data = pd.merge(forests_data, sea_data, on='dates')
    data.set_index('dates', inplace=True)

    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

    features = data_scaled.columns
    for dependent in features:
        for independent in features:
            if dependent != independent:
                clean_data, anomalies = detect_outliers(data_scaled, independent, dependent)
                model, X, y = build_linear_regression(clean_data, independent, dependent)
                plot_regression(X, y, model, independent, dependent, anomalies)


if __name__ == '__main__':
    main()
