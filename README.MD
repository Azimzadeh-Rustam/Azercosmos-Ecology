# Earth observation competition 2024

## Project Overview
Welcome to the official repository for our project submission to the Earth Observation Competition 2024 conducted by [Azercosmos](https://azercosmos.az/en). This project focuses on using advanced satellite imagery analysis and machine learning techniques to analyze the impact of environmental changes in the Caspian Sea on the coastal ecosystems of Azerbaijan.

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

Installation:
```bash
pip3 install numpy==1.24.3 pandas==2.2.2 seaborn==0.13.2 statsmodels==0.14.2 rasterio==1.3.11 matplotlib==3.9.2 scikit-learn==1.5.1 geopandas==1.0.1 tensorflow==2.13.0 torch==2.4.1+cu118 scienceplots
```

## Repository Structure
```
📁 00_src  
    └── (dependence files)  
📁 01_scripts  
    📄 01_kmz_to_geojson.py  
    📄 02_sentinel2_tile.py  
    📄 03_sentinel2_aoi.py  
    📄 04_spectra_reconstruction_neural_network_train.py  
    📄 05_azersky_assemble.py  
    📄 06_segmentation_neural_network_train.py  
    📄 07_visualize_channels.py  
    📄 08_index_calculation.py  
    📄 09_correlation_matrix.py  
    📄 10_linear_regression.py  
📁 02_results
    📁 01_sentinel2
        📁 R20m
            ├── (high-resolution images)  
            ├── (histograms)  
            └── (color maps)  
    📁 02_azersky  
        └── (processed images and results) 
    📁 03_neural_networks
        └── (model outputs and data)
    📁 04_forests_indices  
        └── (calculated indices for forest health)  
    📁 05_sea_indices
        └── (calculated indices for sea health)  
    📁 06_correlations  
        └── (correlation analysis results)  
📄 .gitignore
📝 README.MD
```


## Scripts and Usage

1. **Data Conversion**
   - `01_kmz_to_geojson.py`: Converts KMZ files to GeoJSON format, enabling further geospatial data analysis and usage in GIS applications.

2. **Image Processing**
   - `02_sentinel2_tile.py`: Processes tiles from Sentinel-2 satellite imagery for specific regions, adjusting resolution and alignment.
   - `03_sentinel2_aoi.py`: Extracts and processes Areas of Interest (AOI) from Sentinel-2 imagery for detailed environmental analysis.

3. **Machine Learning Models**
   - `03_spectra_reconstruction_neural_network_train.py`: Trains a neural network to reconstruct missing spectral data, facilitating complete spectral analysis with incomplete datasets.
   - `05_segmentation_neural_network_train.py`: Develops and trains convolutional neural networks for accurate segmentation of satellite images into various land cover types such as forests and water bodies.

4. **Data Assembly**
   - `04_azersky_assemble.py`: Assembles and processes imagery from the Azersky satellite, integrating multiple images to create comprehensive views of targeted areas.

5. **Visualization and Index Calculation**
   - `06_visualize_channels.py`: Provides tools for visualizing different spectral channels of satellite images, aiding in the qualitative assessment of environmental conditions.
   - `07_index_calculation.py`: Calculates environmental indices like NDVI and MNDWI, crucial for assessing vegetation health and water quality.

6. **Statistical Analysis**
   - `08_correlation_matrix.py`: Computes and visualizes correlation matrices to explore relationships between different environmental indices and factors.
   - `09_linear_regression.py`: Applies linear regression analysis to predict environmental impacts based on established correlations, aiding in hypothesis testing and impact assessment.
