"""Build a 0.1-degree building-value density grid for the North Indian Ocean basin.

Values: GEM Global Exposure Model, first-level admin (Adm1) totals of
TOTAL_REPL_COST_USD (building replacement cost + contents).
Boundaries: Natural Earth 1:10m admin-1 states/provinces, joined on ISO 3166-2.
Each admin-1 unit's value is spread uniformly over its land area; value that cannot
be matched to a boundary is spread over the country's unmatched (or whole) area so
that every country total equals GEM's national total."""
import csv, json, math, sys
from collections import defaultdict

GEM_CSV, NE_JSON, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
RES = 0.1
LAT0, LAT1, LON0, LON1 = -5.0, 40.0, 40.0, 105.0
NROWS, NCOLS = round((LAT1 - LAT0) / RES), round((LON1 - LON0) / RES)
R = 6371.0088

# ISO 3166-2 codes that changed between the two sources
ALIAS = {'IN-UT': 'IN-UK', 'IN-OR': 'IN-OD', 'IN-CT': 'IN-CG', 'IN-TG': 'IN-TS'}

def ne_to_gem(code, country):
    code = ALIAS.get(code, code)
    if country == 'LKA' and code.startswith('LK-') and len(code) == 5:
        return code[:4]  # district LK-ab belongs to province LK-a
    return code

gem = defaultdict(dict)
for r in csv.DictReader(open(GEM_CSV, encoding='utf-8')):
    gem[r['ID_0']][r['ID_1']] = {'name': r['NAME_1'], 'usd': float(r['TOTAL_REPL_COST_USD'])}

def ring_area_km2(ring):
    s = 0.0
    for i in range(len(ring) - 1):
        lon1, lat1 = map(math.radians, ring[i]); lon2, lat2 = map(math.radians, ring[i + 1])
        s += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))
    return abs(s * R * R / 2.0)

def polys(g):
    return [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']

feats = []
for f in json.load(open(NE_JSON, encoding='utf-8'))['features']:
    g = f['geometry']
    if not g:
        continue
    pl = polys(g)
    lons = [c[0] for p in pl for c in p[0]]; lats = [c[1] for p in pl for c in p[0]]
    if max(lons) < LON0 or min(lons) > LON1 or max(lats) < LAT0 or min(lats) > LAT1:
        continue
    pr = f['properties']
    country = pr['adm0_a3']
    area = sum(ring_area_km2(p[0]) - sum(ring_area_km2(h) for h in p[1:]) for p in pl)
    feats.append({'country': country, 'code': ne_to_gem(pr['iso_3166_2'], country), 'polys': pl, 'area': area})

# Units: matched GEM admin-1 regions, or a per-country pool for unmatched boundaries
units, unit_idx = [], {}
area_by_unit = defaultdict(float)
for f in feats:
    matched = f['code'] in gem.get(f['country'], {})
    key = (f['country'], f['code'] if matched else '*')
    if key not in unit_idx:
        unit_idx[key] = len(units)
        units.append({'country': f['country'], 'code': key[1],
                      'name': gem[f['country']][f['code']]['name'] if matched else f"{f['country']} (other)"})
    f['unit'] = unit_idx[key]
    area_by_unit[f['unit']] += f['area']

# Value per unit, conserving each country's GEM total
for country in {u['country'] for u in units}:
    regions = gem.get(country, {})
    idxs = [i for i, u in enumerate(units) if u['country'] == country]
    matched_codes = {units[i]['code'] for i in idxs if units[i]['code'] != '*'}
    residual = sum(v['usd'] for k, v in regions.items() if k not in matched_codes)
    pool = [i for i in idxs if units[i]['code'] == '*']
    for i in idxs:
        units[i]['usd'] = regions[units[i]['code']]['usd'] if units[i]['code'] != '*' else 0.0
    if pool:
        units[pool[0]]['usd'] += residual
    elif residual:
        # No unmatched boundary (e.g. a region split after NE was drawn): spread by area
        tot_area = sum(area_by_unit[i] for i in idxs)
        for i in idxs:
            units[i]['usd'] += residual * area_by_unit[i] / tot_area

for i, u in enumerate(units):
    u['usd_per_km2'] = u['usd'] / area_by_unit[i] if area_by_unit[i] else 0.0

# Scanline rasterization at cell centres
grid = [[0] * NCOLS for _ in range(NROWS)]
for f in feats:
    idx = f['unit'] + 1
    for poly in f['polys']:
        edges = [(ring[i], ring[i + 1]) for ring in poly for i in range(len(ring) - 1)]
        plats = [c[1] for c in poly[0]]
        r0 = max(0, int((min(plats) - LAT0) / RES) - 1); r1 = min(NROWS - 1, int((max(plats) - LAT0) / RES) + 1)
        for r in range(r0, r1 + 1):
            y = LAT0 + (r + 0.5) * RES
            xs = sorted(x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                        for (x1, y1), (x2, y2) in edges if (y1 <= y < y2) or (y2 <= y < y1))
            for k in range(0, len(xs) - 1, 2):
                c0 = max(0, math.ceil((xs[k] - LON0) / RES - 0.5)); c1 = min(NCOLS - 1, math.floor((xs[k + 1] - LON0) / RES - 0.5))
                for c in range(c0, c1 + 1):
                    grid[r][c] = idx

used = sorted({v for row in grid for v in row if v})
remap = {old: new + 1 for new, old in enumerate(used)}
out_units = [units[i - 1] for i in used]
rows = []
for row in grid:
    toks, prev, n = [], None, 0
    for v in row:
        v = remap.get(v, 0)
        if v == prev:
            n += 1
        else:
            if prev is not None:
                toks.append(f"{prev:x}" + ('' if n == 1 else f".{n:x}"))
            prev, n = v, 1
    toks.append(f"{prev:x}" + ('' if n == 1 else f".{n:x}"))
    rows.append(','.join(toks))

data = {'lat0': LAT0, 'lon0': LON0, 'res': RES, 'rows': NROWS, 'cols': NCOLS,
        'units': [[u['country'], u['code'], u['name'], round(u['usd_per_km2'])] for u in out_units],
        'rle': rows}
json.dump(data, open(OUT, 'w', encoding='utf-8'), separators=(',', ':'), ensure_ascii=False)
print('units', len(out_units), 'bytes', len(json.dumps(data, separators=(',', ':'), ensure_ascii=False).encode()))
for u in out_units:
    if u['code'] in ('IN-OD', 'IN-WB', 'IN-AP', 'IN-TN', 'BD-C', 'BD-D', 'BD-B', 'MM-07'):
        print(u['code'], u['name'], f"${u['usd']/1e9:.1f}B", f"{u['usd_per_km2']/1e6:.2f}M USD/km2")
