"""Rasterize Natural Earth 50m country polygons into a compact country-ID grid
covering the North Indian Ocean cyclone basin, plus each country's full land
area (km2) computed from its polygon on the sphere."""
import json, math, sys

SRC, OUT, RES = sys.argv[1], sys.argv[2], float(sys.argv[3])
LAT0, LAT1, LON0, LON1 = -5.0, 40.0, 40.0, 105.0
NROWS = round((LAT1 - LAT0) / RES)
NCOLS = round((LON1 - LON0) / RES)
R = 6371.0088

feats = json.load(open(SRC, encoding='utf-8'))['features']

def polys(geom):
    if geom['type'] == 'Polygon':
        return [geom['coordinates']]
    return geom['coordinates']

def ring_area_km2(ring):
    # Spherical polygon area (Chamberlain & Duquette), sign depends on winding
    s = 0.0
    for i in range(len(ring) - 1):
        lon1, lat1 = map(math.radians, ring[i])
        lon2, lat2 = map(math.radians, ring[i + 1])
        s += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))
    return abs(s * R * R / 2.0)

grid = [[0] * NCOLS for _ in range(NROWS)]
codes, names, areas = [], {}, {}

for f in feats:
    p = f['properties']
    code = p.get('ADM0_A3') or p.get('ISO_A3')
    geom = f['geometry']
    if not geom:
        continue
    plist = polys(geom)
    # Full-country land area (outer rings minus holes)
    area = 0.0
    for poly in plist:
        area += ring_area_km2(poly[0]) - sum(ring_area_km2(h) for h in poly[1:])
    # Does the country touch the basin grid?
    lons = [c[0] for poly in plist for c in poly[0]]
    lats = [c[1] for poly in plist for c in poly[0]]
    if max(lons) < LON0 or min(lons) > LON1 or max(lats) < LAT0 or min(lats) > LAT1:
        continue
    if code not in codes:
        codes.append(code)
    idx = codes.index(code) + 1
    names[code] = p.get('NAME')
    areas[code] = round(area)
    # Scanline fill at cell centres (even-odd rule handles holes)
    for poly in plist:
        edges = []
        for ring in poly:
            for i in range(len(ring) - 1):
                edges.append((ring[i], ring[i + 1]))
        plats = [c[1] for c in poly[0]]
        r0 = max(0, int((min(plats) - LAT0) / RES) - 1)
        r1 = min(NROWS - 1, int((max(plats) - LAT0) / RES) + 1)
        for r in range(r0, r1 + 1):
            y = LAT0 + (r + 0.5) * RES
            xs = []
            for (x1, y1), (x2, y2) in edges:
                if (y1 <= y < y2) or (y2 <= y < y1):
                    xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
            xs.sort()
            for k in range(0, len(xs) - 1, 2):
                c0 = max(0, math.ceil((xs[k] - LON0) / RES - 0.5))
                c1 = min(NCOLS - 1, math.floor((xs[k + 1] - LON0) / RES - 0.5))
                for c in range(c0, c1 + 1):
                    grid[r][c] = idx

# Drop countries that ended up with no cells
used = sorted({v for row in grid for v in row if v})
remap = {old: new + 1 for new, old in enumerate(used)}
codes = [codes[i - 1] for i in used]
rows = []
for row in grid:
    # Run-length encode: "id*count" runs joined by commas, base-36 to keep it small
    out, prev, n = [], None, 0
    for v in row:
        v = remap.get(v, 0)
        if v == prev:
            n += 1
        else:
            if prev is not None:
                out.append(f"{prev:x}" + ('' if n == 1 else f".{n:x}"))
            prev, n = v, 1
    out.append(f"{prev:x}" + ('' if n == 1 else f".{n:x}"))
    rows.append(','.join(out))

data = {
    'lat0': LAT0, 'lon0': LON0, 'res': RES, 'rows': NROWS, 'cols': NCOLS,
    'codes': codes,
    'area_km2': {c: areas[c] for c in codes},
    'names': {c: names[c] for c in codes},
    'rle': rows,
}
json.dump(data, open(OUT, 'w'), separators=(',', ':'))
print(len(codes), 'countries:', ' '.join(f"{c}={names[c]}" for c in codes))
print('bytes', len(json.dumps(data, separators=(',', ':'))))
print('areas', {c: areas[c] for c in ['IND', 'BGD', 'MMR', 'LKA', 'PAK'] if c in areas})
