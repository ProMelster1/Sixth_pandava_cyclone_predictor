// Rasterize UNDP grid lines (grid.pmtiles, z6) into km-of-line per 0.1° cell over the basin.
import { PMTiles } from 'pmtiles';
import { VectorTile } from '@mapbox/vector-tile';
import { PbfReader as Pbf } from 'pbf';
import fs from 'fs';
class NodeSource { constructor(p){this.fd=fs.openSync(p,'r');this.p=p} getKey(){return this.p}
  async getBytes(o,l){const b=Buffer.alloc(l);fs.readSync(this.fd,b,0,l,o);return {data:b.buffer.slice(b.byteOffset,b.byteOffset+l)}}}
const pm=new PMTiles(new NodeSource(process.argv[2]));
const LAT0=-5,LAT1=40,LON0=40,LON1=105,RES=0.1,ROWS=450,COLS=650,Z=6;
const km=new Float64Array(ROWS*COLS), kmOsm=new Float64Array(ROWS*COLS);
const n=2**Z;
const lon2x=l=>Math.floor((l+180)/360*n);
const lat2y=l=>Math.floor((1-Math.log(Math.tan(l*Math.PI/180)+1/Math.cos(l*Math.PI/180))/Math.PI)/2*n);
const toLonLat=(tx,ty,px,py,ext)=>{const X=(tx+px/ext)/n, Y=(ty+py/ext)/n;return [X*360-180, Math.atan(Math.sinh(Math.PI*(1-2*Y)))*180/Math.PI];};
const hav=(a,b)=>{const R=6371.0088,r=Math.PI/180;const dLat=(b[1]-a[1])*r,dLon=(b[0]-a[0])*r;const s=Math.sin(dLat/2)**2+Math.cos(a[1]*r)*Math.cos(b[1]*r)*Math.sin(dLon/2)**2;return 2*R*Math.asin(Math.sqrt(s));};
let feats=0,tiles=0,totalKm=0;
for(let tx=lon2x(LON0);tx<=lon2x(LON1-1e-9);tx++){
 for(let ty=lat2y(LAT1);ty<=lat2y(LAT0+1e-9);ty++){
  const t=await pm.getZxy(Z,tx,ty); if(!t||!t.data) continue; tiles++;
  const west=tx/n*360-180, east=(tx+1)/n*360-180;
  const north=Math.atan(Math.sinh(Math.PI*(1-2*ty/n)))*180/Math.PI, south=Math.atan(Math.sinh(Math.PI*(1-2*(ty+1)/n)))*180/Math.PI;
  const vt=new VectorTile(new Pbf(new Uint8Array(t.data))); const layer=vt.layers.grid; if(!layer) continue;
  for(let i=0;i<layer.length;i++){
   const f=layer.feature(i); const osm=f.properties.source==='openstreetmap'; feats++;
   for(const line of f.loadGeometry()){
    for(let k=0;k<line.length-1;k++){
     const a=toLonLat(tx,ty,line[k].x,line[k].y,layer.extent), b=toLonLat(tx,ty,line[k+1].x,line[k+1].y,layer.extent);
     const len=hav(a,b); if(!len) continue;
     const pieces=Math.max(1,Math.ceil(Math.max(Math.abs(b[0]-a[0]),Math.abs(b[1]-a[1]))/0.025));
     for(let p=0;p<pieces;p++){const t2=(p+0.5)/pieces;const lon=a[0]+(b[0]-a[0])*t2, lat=a[1]+(b[1]-a[1])*t2;
      if(lon<west||lon>=east||lat<south||lat>=north) continue;
      const r=Math.floor((lat-LAT0)/RES), c=Math.floor((lon-LON0)/RES); if(r<0||r>=ROWS||c<0||c>=COLS) continue;
      km[r*COLS+c]+=len/pieces; if(osm) kmOsm[r*COLS+c]+=len/pieces; totalKm+=len/pieces;}
    }
   }
  }
 }
}
fs.writeFileSync('km.bin',Buffer.from(km.buffer));fs.writeFileSync('kmosm.bin',Buffer.from(kmOsm.buffer));
let nz=0,max=0;for(const v of km){if(v>0)nz++;if(v>max)max=v;}
const osmKm=kmOsm.reduce((a,b)=>a+b,0);
console.log({tiles,feats,totalKm:Math.round(totalKm),osmShare:(osmKm/totalKm).toFixed(2),nonzeroCells:nz,maxCellKm:Math.round(max)});
// spot checks (km per 0.1° cell)
for(const [name,lat,lon] of [['Kolkata',22.57,88.36],['Bhubaneswar',20.30,85.82],['Puri',19.81,85.83],['Chennai',13.08,80.27],['Dhaka',23.81,90.41],['Bay of Bengal',15.5,88],['Thar desert',27.0,71.0],['Tibet',33,88]]){
 const r=Math.floor((lat-LAT0)/RES),c=Math.floor((lon-LON0)/RES);console.log(name.padEnd(14),km[r*COLS+c].toFixed(1),'km');}
