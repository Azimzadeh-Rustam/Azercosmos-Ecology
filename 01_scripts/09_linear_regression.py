import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np


def build_linear_regression(data, independent, dependent):
    X = sm.add_constant(data[independent])
    y = data[dependent]
    model = sm.OLS(y, X).fit()
    return model, X, y


def plot_regression(X, y, model, independent, dependent):
    plt.figure(figsize=(12, 8))
    plt.scatter(X[independent], y, color='blue', label='Actual Data')  # Реальные данные
    plt.plot(X[independent], model.predict(X), color='red', label='Fitted Line')  # Линия регрессии

    prstd, iv_l, iv_u = wls_prediction_std(model)
    plt.plot(X[independent], iv_l, 'r--', label='Lower Confidence Limit (95%)')
    plt.plot(X[independent], iv_u, 'r--', label='Upper Confidence Limit (95%)')

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
    forests_data = pd.read_json('../03_results/04_correlation_matrix/forests.json', convert_dates=['dates'])
    sea_data = pd.read_json('../03_results/04_correlation_matrix/sea.json', convert_dates=['dates'])
    data = pd.merge(forests_data, sea_data, on='dates')
    data.set_index('dates', inplace=True)

    # Проверить что на оси y должны быть только индексы моря, а на оси x только индексы лесов
    features = data.columns
    for dependent in features:
        for independent in features:
            if dependent != independent:
                model, X, y = build_linear_regression(data, independent, dependent)
                plot_regression(X, y, model, independent, dependent)


if __name__ == '__main__':
    main()
