// Log-quantize km/cell to one base-36 char (0 = no line; level L ≈ 0.05·1.33^(L-1) km) and RLE runs.
import fs from 'fs';
const km=new Float64Array(fs.readFileSync('km.bin').buffer.slice(0));
const ROWS=450,COLS=650,BASE=0.05,F=1.33;
const q=v=>v<=0?0:Math.min(35,Math.max(1,1+Math.round(Math.log(v/BASE)/Math.log(F))));
const deq=L=>L===0?0:BASE*F**(L-1);
const rows=[];let err=0,tot=0;
for(let r=0;r<ROWS;r++){let out='',prev=null,cnt=0;
 const flush=()=>{if(prev===null)return;out+= cnt>3? `${prev}~${cnt.toString(36)}~` : prev.repeat(cnt);};
 for(let c=0;c<COLS;c++){const v=km[r*COLS+c];const L=q(v);err+=deq(L);tot+=v;const ch=L.toString(36);
  if(ch===prev)cnt++;else{flush();prev=ch;cnt=1;}}
 flush();rows.push(out);}
const data={lat0:-5,lon0:40,res:0.1,rows:ROWS,cols:COLS,base:BASE,factor:F,rle:rows};
fs.writeFileSync('power_grid.json',JSON.stringify(data));
console.log('bytes',JSON.stringify(data).length,'total km',Math.round(tot),'decoded total',Math.round(err),'ratio',(err/tot).toFixed(3));
