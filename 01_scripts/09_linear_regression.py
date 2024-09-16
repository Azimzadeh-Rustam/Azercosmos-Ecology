import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np
import scienceplots


FONT_SIZE = 14
MY_FORMATTER = ScalarFormatter(useMathText=True)
MY_FORMATTER.set_scientific(True)
MY_FORMATTER.set_powerlimits((-2, 2))


def set_plot_style():
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'axes.formatter.limits': (-2, 2),
        'axes.formatter.useoffset': True,
        'axes.formatter.offset_threshold': 1
    })


def find_eps(data):
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors.fit(data)
    distances, indices = neighbors.kneighbors(data)

    distances = distances[:, 1]
    return np.percentile(distances, 90)


def detect_outliers(data, eps=0.6, min_samples=2):
    features = data.values

    eps = find_eps(data)

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


def plot_regression(X, y, model, independent, dependent, outliers, save_path):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(MY_FORMATTER)
    ax.yaxis.set_major_formatter(MY_FORMATTER)
    ax.xaxis.get_offset_text().set_size(FONT_SIZE)
    ax.yaxis.get_offset_text().set_size(FONT_SIZE)
    plt.scatter(X[independent], y, color='blue', label='Actual Data')
    plt.plot(X[independent], model.predict(X), color='red', label='Fitted Line')

    prstd, iv_l, iv_u = wls_prediction_std(model)
    plt.plot(X[independent], iv_l, 'r--', label='Lower Confidence Limit (95%)')
    plt.plot(X[independent], iv_u, 'r--', label='Upper Confidence Limit (95%)')

    if not outliers.empty:
        plt.scatter(outliers[independent], outliers[dependent], color='red', s=50, edgecolor='black', alpha=0.65,
                    label='Anomalies')

    pearson_r = np.corrcoef(X[independent], y)[0, 1]
    p_value = model.pvalues.iloc[1]
    std_error = model.bse.iloc[1]

    stats_text = (
        f'Pearson r: {pearson_r:.4f}\n'
        f'P-value: {p_value:.4g}\n'
        f'R-squared: {model.rsquared:.4f}\n'
        f'Standard Error: {std_error:.4f}'
    )
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=FONT_SIZE, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel(f'{independent} normalized value', fontsize=FONT_SIZE)
    plt.ylabel(f'{dependent} normalized value', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=FONT_SIZE)
    plt.title(f'Linear Regression Between\n{dependent} and {independent}', fontsize=FONT_SIZE)
    plt.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    #plt.show()
    plt.savefig(f'{save_path}{dependent}_vs_{independent}.png', dpi=300)
    plt.close()


def main():
    forests_data = pd.read_json('../02_results/04_forests_indices/forests.json', convert_dates=['dates'])
    forests_indices = set(forests_data.columns) - {'dates'}

    sea_data = pd.read_json('../02_results/05_sea_indices/sea.json', convert_dates=['dates'])
    sea_indices = set(sea_data.columns) - {'dates'}

    data = pd.merge(forests_data, sea_data, on='dates')
    data.set_index('dates', inplace=True)

    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

    features = data_scaled.columns
    for dependent in features:
        for independent in features:
            if dependent != independent:
                considered_features = data_scaled[[independent, dependent]]
                clean_data, anomalies = detect_outliers(considered_features)
                model, X, y = build_linear_regression(clean_data, independent, dependent)

                if dependent in forests_indices and independent in sea_indices:
                    save_folder = '../02_results/06_correlations/01_forests_vs_sea/'
                elif dependent in sea_indices and independent in sea_indices:
                    save_folder = '../02_results/06_correlations/02_sea_vs_sea/'
                else:
                    continue

                plot_regression(X, y, model, independent, dependent, anomalies, save_folder)


if __name__ == '__main__':
    set_plot_style()
    main()
