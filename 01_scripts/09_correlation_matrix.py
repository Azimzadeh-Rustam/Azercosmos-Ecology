from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

FONT_SIZE = 14


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


def pairwise_correlation(data):
    columns = data.columns
    n = len(columns)
    correlation_matrix = pd.DataFrame(np.eye(n), columns=columns, index=columns)

    for i in range(n):
        for j in range(i + 1, n):
            feature1 = columns[i]
            feature2 = columns[j]
            considered_features = data[[feature1, feature2]]

            filtered_data, _ = detect_outliers(considered_features)

            if not filtered_data.empty:
                correlation = filtered_data.corr(method='pearson').iloc[0, 1]
                correlation_matrix.loc[feature1, feature2] = correlation
                correlation_matrix.loc[feature2, feature1] = correlation

    return correlation_matrix


def main():
    OUTPUT_PATH = '../02_results/06_correlations/correlation_matrix.png'

    forests_data = pd.read_json('../02_results/04_forests_indices/forests.json', convert_dates=['dates'])
    sea_data = pd.read_json('../02_results/05_sea_indices/sea.json', convert_dates=['dates'])
    data = pd.merge(forests_data, sea_data, on='dates')
    data.set_index('dates', inplace=True)

    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

    correlation_matrix = pairwise_correlation(data_scaled)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, mask=mask, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1,
                cbar=True, square=True, cbar_kws={"shrink": .75}, annot_kws={'fontsize': FONT_SIZE})
    plt.title('Correlation Matrix', fontsize=FONT_SIZE)
    plt.xticks(rotation=45, fontsize=FONT_SIZE, ha='right')
    plt.yticks(fontsize=FONT_SIZE)

    #plt.show()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
