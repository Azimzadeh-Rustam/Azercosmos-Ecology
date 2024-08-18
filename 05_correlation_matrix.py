from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

FONT_SIZE = 12

data = pd.read_json('data.json', convert_dates=['dates'])
data.set_index('dates', inplace=True)

correlation_matrix = data.corr()
mask = np.triu(np.ones_like(correlation_matrix))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, mask=mask, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1,
            cbar=True, square=True, cbar_kws={"shrink": .75}, annot_kws={'fontsize': FONT_SIZE})
plt.title('Correlation Matrix', fontsize=FONT_SIZE)
plt.xticks(rotation=45, fontsize=FONT_SIZE, ha='right')
plt.yticks(fontsize=FONT_SIZE)
plt.show()
plt.close()

criteria_half = len(data.columns) // 2
forest_criteria = data.columns[:criteria_half]
caspian_sea_criteria = data.columns[criteria_half:]

correlation_matrix = pd.DataFrame(index=forest_criteria, columns=caspian_sea_criteria)

for forest_index in forest_criteria:
    for sea_index in caspian_sea_criteria:
        correlation_matrix.at[forest_index, sea_index] = data[forest_index].corr(data[sea_index])

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.astype(float), annot=True, linewidths=0.5, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1,
            cbar=True, square=True, cbar_kws={"shrink": .75}, annot_kws={'fontsize': FONT_SIZE})
plt.title('Correlation Matrix between Forest Health and Caspian Sea Quality')
plt.ylabel('Forest Health Indices')
plt.xlabel('Caspian Sea Quality Indices')
plt.show()
plt.close()
