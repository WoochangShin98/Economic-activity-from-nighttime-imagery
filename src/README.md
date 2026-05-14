# County-Level Nighttime Light Preprocessing

Preprocesses **VIIRS Nighttime Lights (VNL) 2024** satellite data by clipping the global raster to individual US counties and exporting each county as a GeoTIFF and PNG.

## Pipeline

| Step | Description |
|------|-------------|
| 1 | Load & preview global nighttime light raster |
| 2 | Clip to continental USA bounding box |
| 3 | Load US county boundaries (TIGER/Line 2024) |
| 4 | Clip raster per county → `.tif` + `labels.csv` |
| 5 | Render grayscale PNGs with **transparent** background |
| 6 | Render grayscale PNGs with **black** background |

## Data Sources

- **VIIRS VNL 2024 Annual Global** — [eogdata.mines.edu](https://eogdata.mines.edu/nighttime_light/)  
  Filename: `VNL_npp_2024_global_vcmslcfg_v2_c202502261200.average_masked.dat.tif`
- **TIGER/Line US County Shapefile 2024** — [census.gov](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)  
  Folder: `tl_2024_us_county/`

> ⚠️ Raw data files are **not** included in this repository due to size. Download them from the links above and place them in the `data/` folder.

## Setup

```bash
pip install rasterio geopandas shapely matplotlib numpy pandas
```

## Usage

1. Download the raw data files and place them in `data/`
2. Open `County_preprocessing.ipynb`
3. (Optional) Edit paths in **Section 0** if your folder layout differs
4. Run all cells

## Output Structure

```
county_outputs/          # per-county GeoTIFFs + labels.csv
county_overlay_gray_png/ # grayscale PNGs, transparent background
county_dark_png/         # grayscale PNGs, black background
```

## Requirements

- Python 3.8+
- `rasterio`, `geopandas`, `shapely`, `matplotlib`, `numpy`, `pandas`
