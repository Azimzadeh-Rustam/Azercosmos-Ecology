from zipfile import ZipFile
import geopandas as gpd
import io


def kmz_to_geojson(kmz_path: str, output_path: str, target_crs: str = 'EPSG:32639'):
    with ZipFile(kmz_path, 'r') as archived_kml:
        kml_file = next((s for s in archived_kml.namelist() if s.endswith('.kml')), None)

        with archived_kml.open(kml_file) as file:
            with io.BytesIO(file.read()) as data:
                gdf = gpd.read_file(data)

    gdf = gdf.to_crs(target_crs)
    gdf.to_file(output_path, driver='GeoJSON')


def main():
    KMZ_PATH = '../00_src/AOI.kmz'
    GEOJSON_PATH = '../00_src/AOI.geojson'
    kmz_to_geojson(KMZ_PATH, GEOJSON_PATH)


if __name__ == '__main__':
    main()
