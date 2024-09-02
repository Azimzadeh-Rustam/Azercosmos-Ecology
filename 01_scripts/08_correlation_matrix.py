from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

FONT_SIZE = 12


def main():
    forests_data = pd.read_json('../02_results/04_correlation_matrix/forests.json', convert_dates=['dates'])
    sea_data = pd.read_json('../02_results/04_correlation_matrix/sea.json', convert_dates=['dates'])
    data = pd.merge(forests_data, sea_data, on='dates')
    data.set_index('dates', inplace=True)

    correlation_matrix = data.corr(method='pearson')
    mask = np.triu(np.ones_like(correlation_matrix))

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
