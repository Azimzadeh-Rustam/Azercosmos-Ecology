# Earth observation competition 2024

## Project Overview
Welcome to the official repository for our project submission to the Earth Observation Competition 2024. This project focuses on using advanced satellite imagery analysis and machine learning techniques to analyze the impact of environmental changes in the Caspian Sea on the coastal ecosystems of Azerbaijan.

### Description
The project leverages non-invasive remote sensing technologies and AI-driven tools to scrutinize the interconnections between marine and terrestrial ecosystems in the Caspian region. By employing a range of spectral indices and machine learning models, this project aims to:
- Identify and analyze correlations between the quality of the Caspian Sea and the health of adjacent coastal forests.
- Develop mitigation strategies and conservation efforts based on the analyzed data.


## Dependencies

- **Python**: `Version 3.11.7`

### Libraries

- **NumPy**: `Version 1.24.3`
- **Pandas**: `Version 2.2.2`
- **Seaborn**: `Version 0.13.2`
- **Statsmodels**: `Version 0.14.2`
- **Rasterio**: `Version 1.3.11`
- **Matplotlib**: `Version 3.9.2`
- **Scikit-Learn**: `Version 1.5.1`
- **GeoPandas**: `Version 1.0.1`
- **TensorFlow**: `Version 2.13.0`
- **PyTorch**: `Version 2.4.1+cu118`
- **SciencePlots**

## Repository Structure
├── 00_src
│   └── (source code and library files)
├── 01_scripts
│   ├── 01_kmz_to_geojson.py
│   ├── 02_sentinel2_tile.py
│   ├── 03_sentinel2_aoi.py
│   ├── 03_spectra_reconstruction_neural_network_train.py
│   ├── 04_azersky_assemble.py
│   ├── 05_segmentation_neural_network_train.py
│   ├── 06_visualize_channels.py
│   ├── 07_index_calculation.py
│   ├── 08_correlation_matrix.py
│   └── 09_linear_regression.py
└── 02_results
    ├── 01_sentinel2
    │   └── R20m
    │       ├── (high-resolution images)
    │       ├── (histograms)
    │       └── (color maps)
    ├── 02_azersky
    │   └── (processed images and results)
    ├── 03_neural_networks
    │   └── (model outputs and data)
    ├── 04_forests_indices
    │   └── (calculated indices for forest health)
    ├── 05_sea_indices
    │   └── (calculated indices for sea health)
    └── 06_correlations
        └── (correlation analysis results)



