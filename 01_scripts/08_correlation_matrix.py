from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

FONT_SIZE = 12


def detect_outliers(data, eps=0.5, min_samples=2):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(data_scaled)

    labels = model.labels_
    is_outlier = labels == -1

    filtered_data = data[~is_outlier]
    return filtered_data


def main():
    forests_data = pd.read_json('../02_results/04_forests_indices/forests.json', convert_dates=['dates'])
    sea_data = pd.read_json('../02_results/05_sea_indices/sea.json', convert_dates=['dates'])
    data = pd.merge(forests_data, sea_data, on='dates')
    data.set_index('dates', inplace=True)

    data_filtered = detect_outliers(data)

    correlation_matrix = data_filtered.corr(method='pearson')
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(11, 14))
    sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, mask=mask, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1,
                cbar=True, square=True, cbar_kws={"shrink": .75}, annot_kws={'fontsize': FONT_SIZE})
    plt.title('Correlation Matrix', fontsize=FONT_SIZE)
    plt.xticks(rotation=45, fontsize=FONT_SIZE, ha='right')
    plt.yticks(fontsize=FONT_SIZE)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
