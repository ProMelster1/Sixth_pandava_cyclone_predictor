# Rebuilding the dashboard's data grids

The dashboard's impact models read three pre-built grids in `dashboard/assets/`. They are committed, so **you only need this folder to regenerate them**. All three share one grid: 5°S–40°N, 40°E–105°E at 0.1° (~11 km) cells.

| Asset | Used for | Built from |
|---|---|---|
| `country_grid.js` | Civilian exposure (country per cell) | Natural Earth 1:50m countries |
| `value_grid.js` | Projected loss (building value per cell) | GEM Global Exposure Model Adm1 totals + Natural Earth 1:10m states/provinces |
| `power_grid.js` | Grid fragility & outage risk (km of power line per cell) | UNDP global electricity grid (`grid.pmtiles`) |

Run everything from this folder, with the downloads in `raw/` (git-ignored). Needs Python 3.10+ and Node 18+.

## 1. Downloads

```bash
mkdir raw
curl -L -o raw/ne50_countries.geojson https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson
curl -L -o raw/ne10_admin1.geojson   https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson
curl -L -o raw/gem_adm1.csv          https://raw.githubusercontent.com/gem/global_exposure_model/main/World/summaries/Exposure_Summary_Adm1.csv
```

`grid.pmtiles` (~460 MB, UNDP global electricity grid — OpenStreetMap lines plus gridfinder-predicted lines) is too large for the repo. Place it at `raw/grid.pmtiles`.

## 2. Build

```bash
# Country grid (population exposure)
python build_grid.py raw/ne50_countries.geojson country_grid.json 0.1
python wrap_asset.py country_grid.json COUNTRY_GRID ../../dashboard/assets/country_grid.js

# Building-value grid (economic loss)
python build_value_grid.py raw/gem_adm1.csv raw/ne10_admin1.geojson value_grid.json
python wrap_asset.py value_grid.json VALUE_GRID ../../dashboard/assets/value_grid.js

# Power-line grid (outage risk)
cd power_grid
npm install
node extract.mjs ../raw/grid.pmtiles     # writes km.bin (km of line per cell)
node encode.mjs                          # writes power_grid.json
python ../wrap_asset.py power_grid.json POWER_GRID ../../../dashboard/assets/power_grid.js
```

Expected checks: India's building value on the grid ≈ GEM's national total ($3.49T); the basin has ≈1.90M km of power line.
