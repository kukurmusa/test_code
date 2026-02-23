import { useState } from "react";
import { AreaChart, Area, LineChart, Line, BarChart, Bar, ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell, ReferenceLine, ScatterChart, Scatter, ZAxis } from "recharts";

const B="#003366",G="#C4A04A",LB="#93c5fd",SL="#94a3b8",BG="#0f172a",C1="#1e293b",BD="#334155";
const bc=["#64748b","#6366f1","#3b82f6",B];

const Tip=({active,payload,label,fmt})=>{
  if(!active||!payload?.length)return null;
  return(<div style={{background:C1,border:`1px solid ${BD}`,borderRadius:8,padding:"8px 12px",fontSize:11}}>
    <div style={{color:LB,fontWeight:700,marginBottom:3}}>{label}</div>
    {payload.map((p,i)=><div key={i} style={{color:p.color||p.stroke||"#e2e8f0"}}>{p.name}: {fmt?fmt(p.value):p.value}</div>)}
  </div>);
};
const SH=({num,total,title,sub})=>(<div style={{background:`linear-gradient(90deg,${B} 0%,#1e3a5f 100%)`,padding:"14px 32px",borderBottom:`3px solid ${G}`,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
  <div style={{display:"flex",alignItems:"center",gap:14}}><div style={{background:G,width:5,height:32,borderRadius:3}}/><div><div style={{fontSize:10,color:LB,letterSpacing:2,fontWeight:600}}>EMEA CLOSING AUCTION</div><div style={{fontSize:18,fontWeight:700,color:"#fff"}}>{title}</div></div></div>
  <div style={{display:"flex",alignItems:"center",gap:12}}>{sub&&<div style={{fontSize:11,color:SL,maxWidth:200,textAlign:"right"}}>{sub}</div>}<div style={{background:"rgba(196,160,74,0.2)",border:`1px solid ${G}`,borderRadius:6,padding:"4px 10px",fontSize:11,color:G,fontWeight:700}}>{num}/{total}</div></div>
</div>);
const CO=({label,children})=>(<div style={{background:"linear-gradient(90deg,rgba(196,160,74,0.08) 0%,rgba(0,51,102,0.15) 100%)",border:"1px solid rgba(196,160,74,0.25)",borderRadius:8,padding:"12px 16px",display:"flex",gap:12,alignItems:"flex-start",marginTop:12}}>
  <div style={{background:G,color:BG,borderRadius:4,padding:"2px 8px",fontSize:10,fontWeight:800,whiteSpace:"nowrap"}}>{label}</div>
  <div style={{fontSize:12,color:"#cbd5e1",lineHeight:1.5}}>{children}</div>
</div>);
const Src=({t})=><div style={{textAlign:"right",fontSize:9,color:"#475569",marginTop:8}}>{t}</div>;

// ===== DATA =====
const supDem=(()=>{let d=[];for(let p=65;p<=77;p+=.5){let b=p<72?22e6:p<73?22e6-(p-72)*12e6:Math.max(5e5,1e7-(p-73)*22e5);let s=p<70?Math.max(4e5,5e5+(p-65)*1e5):p<72?5e5+(p-65)*1e5+(p-70)*45e5:1e7+(p-72)*24e5;d.push({price:p,buy:Math.round(b),sell:Math.round(s)});}return d;})();
const pqData=(()=>{let d=[];const e={XLON:[5,62,78,82,85,87,88,89,90,91,92,95,100],XPAR:[8,58,72,78,82,85,87,88,89,90,93,96,100],XETR:[3,55,70,76,80,83,85,87,88,89,91,94,99]};[0,25,50,75,100,125,150,175,200,225,250,275,300].forEach((t,i)=>{d.push({t,XLON:e.XLON[i],XPAR:e.XPAR[i],XETR:e.XETR[i]});});return d;})();
const vData=(()=>{let d=[];const ts=[0,10,20,30,40,50,75,100,125,150,175,200,225,250,275,300];const v={XLON:[180,160,120,90,70,55,45,38,32,28,25,22,20,18,22,28],XPAR:[170,148,110,85,65,52,42,35,30,26,23,21,19,17,20,25],XETR:[190,165,125,95,72,58,48,40,34,30,27,24,22,20,24,30]};ts.forEach((t,i)=>{d.push({t,XLON:v.XLON[i],XPAR:v.XPAR[i],XETR:v.XETR[i]});});return d;})();
const phData=(()=>{let d=[];for(let i=0;i<=300;i+=5){let v=i<20?2+i*.5:i<60?12+(i-20)*.15:i<90?18-(i-60)*.2:i<120?12+(i-90)*.05:i<200?13.5-(i-120)*.03:i<260?11-(i-200)*.07:6.8-(i-260)*.1;let p=i<15?i*3:i<60?45+(i-15)*.7:i<200?76.5+(i-60)*.05:i<260?83.5+(i-200)*.15:92.5+(i-260)*.19;d.push({t:i,vol:Math.max(1,v),pq:Math.min(100,p)});}return d;})();
const aLim=(()=>{let d=[];for(let i=0;i<=180;i+=3){let b=i<120?-5+Math.sin(i/20)*8+(Math.sin(i*7.3)*.5)*6:-5+(i-120)*.6+Math.sin(i/10)*3;let s=i<120?-8+Math.cos(i/15)*10+(Math.cos(i*5.1)*.5)*6:-30+(i-120)*.1;if(i>150){b=25+(i-150)*.8;s=-35+(i-150)*.2;}d.push({t:i,buy:Math.round(b),sell:Math.round(s)});}return d;})();
const aImb=(()=>{let d=[];for(let i=0;i<=180;i+=2){let im=i<120?(Math.sin(i*3.7)*.5)*25e3+Math.sin(i/8)*8e3:-15e3+(Math.sin(i*2.1)*.5)*5e3-(i-120)*100;if(i>155)im=-18e3+(Math.cos(i*4.3)*.5)*3e3;d.push({t:i,imb:Math.round(im)});}return d;})();
const sStats=[{bucket:"<25%",corr:.27,r2:.07,beta:.24},{bucket:"<50%",corr:.37,r2:.14,beta:.36},{bucket:"<75%",corr:.61,r2:.38,beta:.57},{bucket:"≤100%",corr:.65,r2:.42,beta:.59}];
const costD=[{pov:"10%",exp:3,real:2},{pov:"20%",exp:6,real:5},{pov:"40%",exp:10,real:12},{pov:"60%",exp:14,real:18},{pov:"80%",exp:19,real:28},{pov:"100%",exp:22,real:38}];
const leakD=(()=>{let d=[];for(let p=150;p<=175;p++){let b=p<158?Math.abs(Math.sin(p))*5e3:p<162?2e4+Math.abs(Math.cos(p*2))*6e4:p<165?1e4+Math.abs(Math.sin(p*3))*15e3:Math.abs(Math.cos(p))*3e3;let s=p<160?Math.abs(Math.cos(p*2))*2e3:p<163?5e3+Math.abs(Math.sin(p))*1e4:p<167?15e3+Math.abs(Math.cos(p*3))*5e4:p<170?8e3+Math.abs(Math.sin(p*2))*5e3:Math.abs(Math.cos(p))*2e3;d.push({price:p,buy:Math.round(b),sell:-Math.round(s)});}return d;})();

// CRS placement data
const crsPrice=(()=>{let d=[];let p=1e3;for(let i=0;i<=60;i++){let dp=i<8?-6+Math.sin(i)*3:i<15?-12+i*.3:i<25?p*.001*Math.sin(i/3):i<35?5+Math.sin(i/2)*3:i<45?10+Math.sin(i/4)*5:15+Math.sin(i/3)*3;p=1e3+dp;d.push({t:i,price:Math.round(p*10)/10,placement:i===8?"P1":i===20?"P2":i===32?"P3":i===38?"P4 (paused)":null});}return d;})();

// Smart Close example data
const scExample=(()=>{let d=[];for(let i=0;i<=300;i+=5){let t=`16:${30+Math.floor(i/60)}:${String(i%60).padStart(2,'0')}`;let indSz=i<30?0:i<60?2e3+i*80:i<150?7e3+i*30:i<250?12e3+i*20:17e3+i*10;let mc=Math.min(indSz*.1,2e3);let tgt=Math.min(indSz*.1,2e3);let opp=i<40?Math.min(800,i*25):i>260?Math.min(1500,(i-260)*40):0;let indPx=i<20?960+i*1.5:i<60?990-Math.sin(i/5)*8:i<150?985+i*.15:i<250?1008+i*.03:1016+Math.sin(i/10)*3;d.push({t:i,indSz:Math.round(indSz),mc:Math.round(mc),tgt:Math.round(tgt),opp:Math.round(opp),indPx:Math.round(indPx*10)/10});}return d;})();

// Excess volume data
const exVolData=(()=>{let d=[];for(let x=0;x<=30;x+=2){let v=x<5?x*.5:x<10?2.5+(x-5)*1.2:x<20?8.5+(x-10)*.8:16.5+(x-20)*.3;d.push({lim:x/10,vol1:Math.round(v*10)/10,vol2:Math.round(v*1.15*10)/10,vol3:Math.round(v*1.3*10)/10,vol4:Math.round(v*1.5*10)/10,impact1:Math.round((x/10)*.8*10)/10,impact2:Math.round((x/10)*.75*10)/10,impact3:Math.round((x/10)*.7*10)/10,impact4:Math.round((x/10)*.65*10)/10});}return d;})();
const exVolBats=(()=>{let d=[];for(let x=0;x<=30;x+=2){let v=x<8?x*.3:x<14?2.4+(x-8)*2.5:x<20?17.4+(x-14)*.8:22.2+(x-20)*.2;d.push({lim:x/10,vol:Math.round(v*10)/10});}return d;})();

const phC=["#64748b","#ef4444","#f97316","#eab308","#22c55e","#3b82f6"];
const phL=["Stale Orders","Price Discovery","Convergence","Steady State","Angels Arrive","Final Pairing"];
const phDesc=["Residual orders from continuous session create initial dislocation","New orders arrive rapidly — indicative price is most volatile","Book deepens, indicative price begins converging","Stable volatility, paired quantity largely unchanged","'Auction Angels' provide liquidity, price snaps toward fair value","Final 25% of volume pairs — last chance for adjustments"];

// ===== SLIDES =====
const S1=()=>(<div style={{padding:"16px 28px"}}><div style={{display:"grid",gridTemplateColumns:"340px 1fr",gap:24}}>
  <div><div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>How the Close Price is Set</div>
    <div style={{fontSize:12,color:SL,lineHeight:1.7,marginBottom:12}}>European closing auctions determine the close price by finding the price that <strong style={{color:G}}>maximizes traded volume</strong> while <strong style={{color:G}}>minimizing imbalance</strong>.</div>
    <div style={{fontSize:12,color:SL,lineHeight:1.7,marginBottom:12}}>During the 5-minute call phase, orders accumulate with no matching. At the random auction end, all orders clear at a single price.</div>
    <div style={{background:C1,borderRadius:8,padding:12,border:`1px solid ${BD}`,marginBottom:10}}><div style={{fontSize:11,fontWeight:700,color:LB,marginBottom:6}}>Price Priority Rules</div><div style={{fontSize:11,color:SL,lineHeight:1.6}}>1. Maximize executable volume<br/>2. Minimize remaining imbalance<br/>3. Minimize distance to reference price</div></div>
    <div style={{fontSize:11,color:SL}}>Example: VOD LN, 29 Apr 2025 — L3 aggregated book just before uncrossing.</div></div>
  <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}>
    <div style={{fontSize:12,fontWeight:700,color:LB,marginBottom:2}}>VOD LN | L3 Aggregated Book at 16:35:14</div>
    <ResponsiveContainer width="100%" height={300}><AreaChart data={supDem} margin={{top:5,right:10,left:10,bottom:0}}><CartesianGrid strokeDasharray="3 3" stroke={BD}/><XAxis dataKey="price" tick={{fill:SL,fontSize:10}} label={{value:"Price",position:"insideBottom",offset:-2,fill:"#64748b",fontSize:10}}/><YAxis tick={{fill:SL,fontSize:10}} tickFormatter={v=>(v/1e6).toFixed(0)+"M"}/><Tooltip content={<Tip fmt={v=>(v/1e6).toFixed(1)+"M"}/>}/><Area type="stepAfter" dataKey="buy" stroke="#3b82f6" fill="rgba(59,130,246,0.2)" strokeWidth={2} name="Buy"/><Area type="stepAfter" dataKey="sell" stroke="#f97316" fill="rgba(249,115,22,0.15)" strokeWidth={2} name="Sell"/><ReferenceLine x={72} stroke={G} strokeDasharray="6 3" strokeWidth={2} label={{value:"Clearing: 72.0",position:"top",fill:G,fontSize:11,fontWeight:700}}/><Legend wrapperStyle={{fontSize:10}}/></AreaChart></ResponsiveContainer>
    <div style={{textAlign:"center",marginTop:4,fontSize:11,color:G,fontWeight:600}}>▲ Clearing price at intersection — ~10M shares matched</div></div>
</div><CO label="KEY">Every order you submit changes these curves — and therefore the clearing price. Understanding your share of the auction is critical.</CO><Src t="Source: RBC Capital Markets / L3 Market Data"/></div>);

const S2=()=>(<div style={{padding:"16px 28px"}}><div style={{fontSize:12,color:SL,marginBottom:14}}>Volume pairs rapidly early, while volatility decays continuously. Understanding this timing is essential for order placement.</div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
    <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}><div style={{fontSize:12,fontWeight:700,color:LB,marginBottom:6}}>Paired Quantity (% of Final Volume)</div>
      <ResponsiveContainer width="100%" height={200}><LineChart data={pqData} margin={{top:5,right:10,left:0,bottom:0}}><CartesianGrid strokeDasharray="3 3" stroke={BD}/><XAxis dataKey="t" tick={{fill:SL,fontSize:10}} label={{value:"Seconds",position:"insideBottom",offset:-2,fill:"#64748b",fontSize:10}}/><YAxis tick={{fill:SL,fontSize:10}} domain={[0,100]} tickFormatter={v=>v+"%"}/><Tooltip content={<Tip fmt={v=>v+"%"}/>}/><Line type="monotone" dataKey="XLON" stroke="#3b82f6" strokeWidth={2} dot={false} name="LSE"/><Line type="monotone" dataKey="XPAR" stroke="#f97316" strokeWidth={2} dot={false} name="Euronext"/><Line type="monotone" dataKey="XETR" stroke="#22c55e" strokeWidth={2} dot={false} name="Xetra"/><ReferenceLine x={50} stroke={G} strokeDasharray="4 4" label={{value:"~75%",position:"insideTopRight",fill:G,fontSize:10}}/><Legend wrapperStyle={{fontSize:10}}/></LineChart></ResponsiveContainer>
      <div style={{display:"flex",justifyContent:"center",gap:20,marginTop:4}}><div style={{textAlign:"center"}}><div style={{fontSize:18,fontWeight:800,color:"#3b82f6"}}>75%</div><div style={{fontSize:9,color:SL}}>paired by 50s</div></div><div style={{textAlign:"center"}}><div style={{fontSize:18,fontWeight:800,color:G}}>25%</div><div style={{fontSize:9,color:SL}}>in final 30s</div></div></div></div>
    <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}><div style={{fontSize:12,fontWeight:700,color:LB,marginBottom:6}}>Indicative Price Volatility (bps)</div>
      <ResponsiveContainer width="100%" height={200}><ComposedChart data={vData} margin={{top:5,right:10,left:0,bottom:0}}><CartesianGrid strokeDasharray="3 3" stroke={BD}/><XAxis dataKey="t" tick={{fill:SL,fontSize:10}} label={{value:"Seconds",position:"insideBottom",offset:-2,fill:"#64748b",fontSize:10}}/><YAxis tick={{fill:SL,fontSize:10}}/><Tooltip content={<Tip fmt={v=>v+" bps"}/>}/><Line type="monotone" dataKey="XLON" stroke="#3b82f6" strokeWidth={2} dot={false} name="LSE"/><Line type="monotone" dataKey="XPAR" stroke="#f97316" strokeWidth={2} dot={false} name="Euronext"/><Line type="monotone" dataKey="XETR" stroke="#22c55e" strokeWidth={2} dot={false} name="Xetra"/><Legend wrapperStyle={{fontSize:10}}/></ComposedChart></ResponsiveContainer>
      <div style={{display:"flex",justifyContent:"center",gap:20,marginTop:4}}><div style={{textAlign:"center"}}><div style={{fontSize:18,fontWeight:800,color:"#ef4444"}}>180</div><div style={{fontSize:9,color:SL}}>bps at start</div></div><div style={{textAlign:"center"}}><div style={{fontSize:18,fontWeight:800,color:"#22c55e"}}>~20</div><div style={{fontSize:9,color:SL}}>bps at stable</div></div></div></div>
  </div><CO label="TIMING">Early orders face high volatility but attract reactive liquidity. Late orders face low volatility but less absorption time. Optimal timing balances these forces.</CO><Src t="Source: RBC Capital Markets / Market Data | Jan–Nov 2024"/></div>);

const S3=()=>(<div style={{padding:"16px 28px"}}><div style={{fontSize:12,color:SL,marginBottom:12}}>We identify <strong style={{color:"#fff"}}>6 distinct phases</strong> during the call period. Each requires a different trading approach.</div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 280px",gap:16}}>
    <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}>
      <ResponsiveContainer width="100%" height={260}><ComposedChart data={phData} margin={{top:10,right:40,left:0,bottom:0}}><CartesianGrid strokeDasharray="3 3" stroke={BD}/><XAxis dataKey="t" tick={{fill:SL,fontSize:10}} ticks={[0,60,120,180,240,300]} tickFormatter={v=>`16:${31+Math.floor(v/60)}`}/><YAxis yAxisId="l" tick={{fill:SL,fontSize:10}} domain={[0,20]} label={{value:"Vol %",angle:-90,position:"insideLeft",fill:"#64748b",fontSize:10,dx:8}}/><YAxis yAxisId="r" orientation="right" tick={{fill:SL,fontSize:10}} domain={[0,100]} tickFormatter={v=>v+"%"}/><Tooltip content={<Tip/>}/><Area yAxisId="r" type="monotone" dataKey="pq" fill="rgba(59,130,246,0.1)" stroke="none"/><Line yAxisId="r" type="monotone" dataKey="pq" stroke="#3b82f6" strokeWidth={2} dot={false} name="Paired %"/><Line yAxisId="l" type="monotone" dataKey="vol" stroke={G} strokeWidth={2.5} dot={false} name="Volatility"/>{[0,20,60,120,200,260].map((x,i)=><ReferenceLine key={i} yAxisId="l" x={x} stroke={phC[i]} strokeDasharray="4 3" strokeOpacity={.5}/>)}<Legend wrapperStyle={{fontSize:10}}/></ComposedChart></ResponsiveContainer></div>
    <div style={{display:"flex",flexDirection:"column",gap:5}}>{phL.map((l,i)=>(<div key={i} style={{background:C1,borderRadius:7,padding:"6px 10px",border:`1px solid ${BD}`,borderLeft:`3px solid ${phC[i]}`}}><div style={{display:"flex",gap:6,marginBottom:1}}><span style={{fontSize:9,fontWeight:800,color:phC[i]}}>P{i+1}</span><span style={{fontSize:11,fontWeight:700,color:"#fff"}}>{l}</span></div><div style={{fontSize:9,color:SL,lineHeight:1.4}}>{phDesc[i]}</div></div>))}</div>
  </div><CO label="FRAMEWORK">Phases 5-6 — "Auction Angels" — are where the price is finalized. Our algo's intelligence concentrates here.</CO><Src t="Source: LSE / BMLL Technologies | Jan–Nov 2024"/></div>);

const S4=()=>(<div style={{padding:"16px 28px"}}><div style={{fontSize:12,color:SL,marginBottom:12}}>In the <strong style={{color:"#fff"}}>final 20 seconds</strong>, "Auction Angels" submit aggressive limit orders, shifting the imbalance and moving the indicative price toward fair value.</div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
    <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}><div style={{fontSize:12,fontWeight:700,color:LB}}>Aggressive Limit Orders (% Change)</div><div style={{fontSize:10,color:"#475569",marginBottom:6}}>AIR FP — 17 Jan 2025</div>
      <ResponsiveContainer width="100%" height={210}><LineChart data={aLim} margin={{top:5,right:10,left:0,bottom:0}}><CartesianGrid strokeDasharray="3 3" stroke={BD}/><XAxis dataKey="t" tick={{fill:SL,fontSize:9}}/><YAxis tick={{fill:SL,fontSize:10}}/><Tooltip content={<Tip fmt={v=>v+"%"}/>}/><Line type="monotone" dataKey="buy" stroke="#22c55e" strokeWidth={2} dot={false} name="Buy"/><Line type="monotone" dataKey="sell" stroke="#ef4444" strokeWidth={2} dot={false} name="Sell"/><ReferenceLine x={140} stroke={G} strokeDasharray="4 3" label={{value:"Last 20s →",fill:G,fontSize:10}}/><Legend wrapperStyle={{fontSize:10}}/></LineChart></ResponsiveContainer></div>
    <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}><div style={{fontSize:12,fontWeight:700,color:LB}}>L1 Imbalance (# Shares)</div><div style={{fontSize:10,color:"#475569",marginBottom:6}}>Imbalance shifts sell-heavy despite buy aggression</div>
      <ResponsiveContainer width="100%" height={210}><BarChart data={aImb} margin={{top:5,right:10,left:0,bottom:0}}><CartesianGrid strokeDasharray="3 3" stroke={BD}/><XAxis dataKey="t" tick={{fill:SL,fontSize:9}}/><YAxis tick={{fill:SL,fontSize:10}} tickFormatter={v=>(v/1e3).toFixed(0)+"K"}/><Tooltip content={<Tip fmt={v=>(v/1e3).toFixed(1)+"K"}/>}/><Bar dataKey="imb" name="Imbalance">{aImb.map((e,i)=><Cell key={i} fill={e.imb>0?"#22c55e":"#3b82f6"} opacity={.7}/>)}</Bar><ReferenceLine y={0} stroke="#475569"/><ReferenceLine x={140} stroke={G} strokeDasharray="4 3"/></BarChart></ResponsiveContainer></div>
  </div><CO label="INSIGHT">Auction Angels exploit imbalances — offsetting flow earns a positive cost advantage. Our algo monitors this in real time.</CO><Src t="Source: RBC Capital Markets / L3 Market Data — AIR FP"/></div>);

const S5=()=>(<div style={{padding:"16px 28px"}}><div style={{fontSize:12,color:SL,marginBottom:14}}>Three structural challenges affect close auction execution quality.</div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14,marginBottom:14}}>
    {[{n:"01",t:"Your Limit IS the Price",c:"#ef4444",d:"At high POV, parent limit directly determines clearing price",s:"6×",sd:"R² increase"},
      {n:"02",t:"Visible = Vulnerable",c:"#f97316",d:"L3 transparency means large orders leak intent to the market",s:"L3",sd:"full visibility"},
      {n:"03",t:"Static Models Fail",c:"#eab308",d:"Pre-trade models underestimate cost by up to 2× at high POV",s:"2×",sd:"cost gap"}
    ].map((c,i)=>(<div key={i} style={{background:C1,borderRadius:10,padding:14,border:`1px solid ${BD}`,borderTop:`3px solid ${c.c}`}}>
      <div style={{fontSize:10,fontWeight:800,color:c.c}}>CHALLENGE {c.n}</div><div style={{fontSize:13,fontWeight:700,color:"#fff",margin:"4px 0"}}>{c.t}</div><div style={{fontSize:11,color:SL,lineHeight:1.5}}>{c.d}</div>
      <div style={{textAlign:"center",marginTop:10,paddingTop:8,borderTop:`1px solid ${BD}`}}><span style={{fontSize:24,fontWeight:800,color:c.c}}>{c.s}</span><div style={{fontSize:9,color:"#64748b"}}>{c.sd}</div></div></div>))}
  </div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
    <div style={{background:C1,borderRadius:10,padding:"12px 12px 4px",border:`1px solid ${BD}`}}><div style={{fontSize:11,fontWeight:700,color:LB,marginBottom:4}}>R² by POV Bucket</div>
      <ResponsiveContainer width="100%" height={140}><BarChart data={sStats} margin={{top:5,right:10,left:0,bottom:0}}><CartesianGrid strokeDasharray="3 3" stroke={BD}/><XAxis dataKey="bucket" tick={{fill:SL,fontSize:10}}/><YAxis tick={{fill:SL,fontSize:10}} domain={[0,.5]}/><Bar dataKey="r2" name="R²" radius={[4,4,0,0]} barSize={28}>{sStats.map((e,i)=><Cell key={i} fill={bc[i]}/>)}</Bar></BarChart></ResponsiveContainer></div>
    <div style={{background:C1,borderRadius:10,padding:"12px 12px 4px",border:`1px solid ${BD}`}}><div style={{fontSize:11,fontWeight:700,color:LB,marginBottom:4}}>Expected vs Realised Cost (bps)</div>
      <ResponsiveContainer width="100%" height={140}><BarChart data={costD} margin={{top:5,right:10,left:0,bottom:0}}><CartesianGrid strokeDasharray="3 3" stroke={BD}/><XAxis dataKey="pov" tick={{fill:SL,fontSize:10}}/><YAxis tick={{fill:SL,fontSize:10}}/><Bar dataKey="exp" name="Expected" fill="#3b82f6" barSize={12} radius={[2,2,0,0]}/><Bar dataKey="real" name="Realised" fill="#f97316" barSize={12} radius={[2,2,0,0]}/><Legend wrapperStyle={{fontSize:10}}/></BarChart></ResponsiveContainer></div>
  </div><CO label="NEXT">These challenges define the design requirements for Smart Close.</CO></div>);

const S6=()=>(<div style={{padding:"16px 28px"}}><div style={{fontSize:12,color:SL,marginBottom:14}}>Large orders on the transparent European book leak intent. Understanding where limits cluster and what the market sees is critical.</div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
    <div style={{background:C1,borderRadius:10,padding:14,border:`1px solid ${BD}`}}>
      <div style={{fontSize:12,fontWeight:700,color:LB,marginBottom:6}}>Where the Market Sets Limits</div>
      <div style={{display:"flex",justifyContent:"center",alignItems:"center",height:190,background:"rgba(59,130,246,0.05)",borderRadius:8,border:`1px dashed ${BD}`}}>
        <div style={{textAlign:"center"}}><div style={{fontSize:36,fontWeight:800,color:"#3b82f6"}}>300–500</div><div style={{fontSize:12,color:SL}}>bps through last continuous price</div><div style={{fontSize:10,color:"#64748b",marginTop:6}}>Consistent across STOXX50 L3 and RBC orders</div></div></div></div>
    <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}><div style={{fontSize:12,fontWeight:700,color:LB,marginBottom:6}}>Information Leakage — What the Market Sees</div>
      <ResponsiveContainer width="100%" height={190}><BarChart data={leakD} margin={{top:5,right:10,left:0,bottom:0}} stackOffset="sign"><CartesianGrid strokeDasharray="3 3" stroke={BD}/><XAxis dataKey="price" tick={{fill:SL,fontSize:9}}/><YAxis tick={{fill:SL,fontSize:9}} tickFormatter={v=>Math.abs(v/1e3).toFixed(0)+"K"}/><ReferenceLine y={0} stroke="#475569"/><Bar dataKey="buy" name="Buy" fill="#22c55e" stackId="s" barSize={10}/><Bar dataKey="sell" name="Sell" fill="#ef4444" stackId="s" barSize={10}/><ReferenceLine x={162} stroke={G} strokeDasharray="4 3" label={{value:"IMP",fill:G,fontSize:9,position:"top"}}/><Legend wrapperStyle={{fontSize:10}}/></BarChart></ResponsiveContainer></div>
  </div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14,marginTop:12}}>
    <CO label="PROBLEM">80K shares at one level is a neon sign. Every participant trades against you.</CO>
    <CO label="SOLUTION">Our algo slices across levels automatically, calibrated to real-time book depth.</CO>
  </div><Src t="Source: RBC Capital Markets / STOXX50 L3 Data"/></div>);

const S7=()=>(<div style={{padding:"16px 28px"}}><div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20,marginBottom:12}}>
  <div><div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:8}}>Expected Close Cost Model</div>
    <div style={{background:C1,borderRadius:10,padding:14,border:`1px solid ${BD}`,marginBottom:10}}>
      <div style={{fontFamily:"monospace",fontSize:13,color:"#e2e8f0",textAlign:"center",lineHeight:2}}><span style={{color:G}}>E[cost]</span> = <span style={{color:"#64748b"}}>c₁/(1-c₄)</span> · <span style={{color:"#3b82f6"}}>σ<sup>c₂</sup></span> · <span style={{color:"#22c55e"}}>(q/moc)<sup>c₃</sup></span> · <span style={{color:"#f97316"}}>(moc/ADV)<sup>1-c₄</sup></span></div></div>
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>{[{s:"σ",l:"Auction volatility",c:"#3b82f6"},{s:"q/moc",l:"Size ÷ close vol",c:"#22c55e"},{s:"moc/ADV",l:"Close ÷ daily vol",c:"#f97316"},{s:"c₁₋₄",l:"Calibrated params",c:"#64748b"}].map((v,i)=>(<div key={i} style={{background:C1,borderRadius:6,padding:"5px 8px",border:`1px solid ${BD}`,display:"flex",gap:6,alignItems:"center"}}><span style={{fontFamily:"monospace",fontSize:13,fontWeight:700,color:v.c}}>{v.s}</span><span style={{fontSize:10,color:SL}}>{v.l}</span></div>))}</div></div>
  <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}><div style={{fontSize:12,fontWeight:700,color:LB,marginBottom:6}}>Expected vs Realised Cost</div>
    <ResponsiveContainer width="100%" height={200}><BarChart data={costD} margin={{top:10,right:10,left:0,bottom:0}}><CartesianGrid strokeDasharray="3 3" stroke={BD}/><XAxis dataKey="pov" tick={{fill:SL,fontSize:11}} label={{value:"Close POV %",position:"insideBottom",offset:-2,fill:"#64748b",fontSize:10}}/><YAxis tick={{fill:SL,fontSize:11}}/><Tooltip content={<Tip fmt={v=>v+" bps"}/>}/><Bar dataKey="exp" name="Expected" fill="#3b82f6" barSize={16} radius={[3,3,0,0]}/><Bar dataKey="real" name="Realised" fill="#f97316" barSize={16} radius={[3,3,0,0]}/><Legend wrapperStyle={{fontSize:10}}/></BarChart></ResponsiveContainer>
    <div style={{textAlign:"center",fontSize:11,color:"#ef4444",fontWeight:700}}>▲ At 100% POV: Realised ≈ 2× Expected</div></div>
</div>
<div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10}}>
  {[{i:"✓",t:"Good For",d:"Basket cost estimates, trend analysis",c:"#22c55e"},{i:"✗",t:"Insufficient For",d:"Intra-auction decisions — misses live conditions",c:"#ef4444"},{i:"→",t:"Our Approach",d:"Combine pre-trade + real-time L2/L3 signals",c:G}].map((c,j)=>(<div key={j} style={{background:C1,borderRadius:8,padding:10,border:`1px solid ${BD}`}}><div style={{display:"flex",gap:6,alignItems:"center",marginBottom:3}}><span style={{fontSize:14,color:c.c}}>{c.i}</span><span style={{fontSize:11,fontWeight:700,color:c.c}}>{c.t}</span></div><div style={{fontSize:10,color:SL,lineHeight:1.5}}>{c.d}</div></div>))}
</div><Src t="Source: RBC Capital Markets / Execution Data"/></div>);

// NEW SLIDES: Algo Design
const S8=()=>(<div style={{padding:"16px 28px"}}>
  <div style={{fontSize:12,color:SL,marginBottom:16}}>Our research identifies three questions a close algo must answer. <strong style={{color:"#fff"}}>Smart Close</strong> addresses each through the Close Response System (CRS).</div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
    <div>
      <div style={{fontSize:13,fontWeight:700,color:"#fff",marginBottom:12}}>Key Design Questions</div>
      {[{q:"How to trade large orders without overly impacting the market?",a:"Place-and-review mechanism with live impact monitoring",c:"#ef4444",icon:"⚡"},
        {q:"How to reduce information leakage on the transparent book?",a:"Automatic order slicing across multiple price levels",c:"#f97316",icon:"🔒"},
        {q:"How to dynamically evaluate conditions and adapt?",a:"Close Response System (CRS) with real-time L2 orderbook signals",c:"#22c55e",icon:"📡"}
      ].map((r,i)=>(<div key={i} style={{background:C1,borderRadius:10,padding:14,border:`1px solid ${BD}`,borderLeft:`3px solid ${r.c}`,marginBottom:10}}>
        <div style={{fontSize:12,fontWeight:700,color:"#fff",marginBottom:4}}>{r.icon} {r.q}</div>
        <div style={{fontSize:11,color:r.c,fontWeight:600}}>→ {r.a}</div></div>))}
    </div>
    <div style={{background:C1,borderRadius:10,padding:16,border:`1px solid ${BD}`}}>
      <div style={{fontSize:13,fontWeight:700,color:LB,marginBottom:12}}>RBC Smart Close Architecture</div>
      <div style={{background:"rgba(0,51,102,0.3)",borderRadius:8,padding:14,border:`1px solid ${B}`,marginBottom:12}}>
        <div style={{fontSize:12,fontWeight:700,color:G,textAlign:"center",marginBottom:8}}>Close Response System (CRS)</div>
        <div style={{fontSize:11,color:SL,textAlign:"center",lineHeight:1.5}}>Place → Monitor → Evaluate Impact → Decide → Repeat</div>
      </div>
      {["Monitors L2 orderbook changes at set intervals","Evaluates cost surprise from each placement using live impact model","Decides: keep, cancel, or partially cancel existing orders","Places new slices after the book stabilizes","Repeats until auction uncross"].map((s,i)=>(<div key={i} style={{display:"flex",gap:8,alignItems:"flex-start",marginBottom:6}}>
        <div style={{minWidth:20,height:20,borderRadius:10,background:i<4?["#ef4444","#f97316","#eab308","#22c55e"][i]:"#3b82f6",display:"flex",alignItems:"center",justifyContent:"center",fontSize:10,fontWeight:800,color:"#fff"}}>{i+1}</div>
        <div style={{fontSize:11,color:SL,lineHeight:1.4}}>{s}</div></div>))}
      <div style={{marginTop:10,padding:"8px 10px",background:"rgba(196,160,74,0.1)",borderRadius:6,border:`1px solid rgba(196,160,74,0.2)`}}>
        <div style={{fontSize:10,color:G,fontWeight:600}}>Orders broken into smaller chunks to reduce footprint on the L3 book</div></div>
    </div>
  </div><CO label="KEY">Smart Close turns our microstructure research into a systematic, adaptive execution framework — balancing market impact, information leakage, and liquidity capture in real time.</CO></div>);

const S9=()=>(<div style={{padding:"16px 28px"}}>
  <div style={{fontSize:12,color:SL,marginBottom:6}}>The CRS uses a <strong style={{color:"#fff"}}>place-and-review loop</strong> to manage market impact dynamically. Each placement is evaluated against a live cost surprise metric.</div>
  <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`,marginBottom:14}}>
    <div style={{fontSize:12,fontWeight:700,color:LB,marginBottom:6}}>CRS Decision Flow — Illustrative Example</div>
    <ResponsiveContainer width="100%" height={200}>
      <ComposedChart data={crsPrice} margin={{top:20,right:20,left:0,bottom:0}}>
        <CartesianGrid strokeDasharray="3 3" stroke={BD}/>
        <XAxis dataKey="t" tick={{fill:SL,fontSize:10}} label={{value:"Time (auction seconds)",position:"insideBottom",offset:-2,fill:"#64748b",fontSize:10}}/>
        <YAxis tick={{fill:SL,fontSize:10}} domain={['auto','auto']} label={{value:"Price",angle:-90,position:"insideLeft",fill:"#64748b",fontSize:10,dx:8}}/>
        <Tooltip content={<Tip/>}/>
        <Area type="monotone" dataKey="price" fill="rgba(196,160,74,0.08)" stroke="none"/>
        <Line type="monotone" dataKey="price" stroke={G} strokeWidth={2.5} dot={false} name="Indicative Price"/>
        {[{x:8,l:"P1: Place",c:"#22c55e"},{x:20,l:"P2: No surprise → place",c:"#22c55e"},{x:32,l:"P3: Surprise → cancel",c:"#ef4444"},{x:38,l:"P4: Paused",c:"#f97316"}].map((p,i)=>(
          <ReferenceLine key={i} x={p.x} stroke={p.c} strokeDasharray="4 3" label={{value:p.l,position:"top",fill:p.c,fontSize:9,fontWeight:600}}/>
        ))}
      </ComposedChart>
    </ResponsiveContainer></div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:10}}>
    {[{step:"Place",desc:"Submit slice into the auction book",icon:"📤",c:"#22c55e"},
      {step:"Monitor",desc:"Track L2 book changes and price reaction",icon:"👁",c:"#3b82f6"},
      {step:"Evaluate",desc:"Compare actual impact vs. expected from cost model",icon:"📊",c:"#f97316"},
      {step:"Decide",desc:"Keep if no surprise, cancel if cost exceeds threshold",icon:"⚖️",c:"#ef4444"}
    ].map((s,i)=>(<div key={i} style={{background:C1,borderRadius:8,padding:10,border:`1px solid ${BD}`,textAlign:"center"}}>
      <div style={{fontSize:20,marginBottom:4}}>{s.icon}</div>
      <div style={{fontSize:12,fontWeight:700,color:s.c,marginBottom:2}}>{s.step}</div>
      <div style={{fontSize:10,color:SL,lineHeight:1.4}}>{s.desc}</div></div>))}
  </div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14,marginTop:12}}>
    <CO label="IMPACT">Live cost model replaces static historical estimates — adapts to today's specific auction conditions.</CO>
    <CO label="SLICING">Each placement is broken across multiple price levels, reducing the information footprint visible on the L3 book.</CO>
  </div></div>);

const S10=()=>(<div style={{padding:"16px 28px"}}>
  <div style={{fontSize:12,color:SL,marginBottom:10}}>Example: A buy order that ultimately accounts for <strong style={{color:"#fff"}}>80% of market close volume</strong>. The strategy manages three distinct quantity tiers.</div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20,marginBottom:12}}>
    <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}>
      <div style={{fontSize:12,fontWeight:700,color:LB,marginBottom:6}}>Order Quantities vs Market Indicative Size</div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={scExample} margin={{top:5,right:10,left:0,bottom:0}}>
          <CartesianGrid strokeDasharray="3 3" stroke={BD}/>
          <XAxis dataKey="t" tick={{fill:SL,fontSize:9}} ticks={[0,60,120,180,240,300]} tickFormatter={v=>`16:${30+Math.floor(v/60)}`}/>
          <YAxis yAxisId="l" tick={{fill:SL,fontSize:9}} tickFormatter={v=>(v/1e3).toFixed(0)+"K"} label={{value:"Shares",angle:-90,position:"insideLeft",fill:"#64748b",fontSize:9,dx:8}}/>
          <YAxis yAxisId="r" orientation="right" tick={{fill:SL,fontSize:9}} tickFormatter={v=>(v/1e3).toFixed(0)+"K"}/>
          <Tooltip content={<Tip fmt={v=>Math.round(v).toLocaleString()}/>}/>
          <Area yAxisId="l" type="monotone" dataKey="indSz" fill="rgba(148,163,184,0.1)" stroke={SL} strokeWidth={1} strokeDasharray="4 3" name="Market Size"/>
          <Area yAxisId="r" type="monotone" dataKey="mc" fill="rgba(59,130,246,0.4)" stroke="#3b82f6" strokeWidth={1.5} stackId="q" name="Must Complete (10%)"/>
          <Area yAxisId="r" type="monotone" dataKey="tgt" fill="rgba(249,115,22,0.4)" stroke="#f97316" strokeWidth={1.5} stackId="q" name="Target (10%)"/>
          <Area yAxisId="r" type="monotone" dataKey="opp" fill="rgba(34,197,94,0.4)" stroke="#22c55e" strokeWidth={1.5} stackId="q" name="Opportunistic (60%)"/>
          <Legend wrapperStyle={{fontSize:9}}/>
        </ComposedChart>
      </ResponsiveContainer></div>
    <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}>
      <div style={{fontSize:12,fontWeight:700,color:LB,marginBottom:6}}>Indicative Price Evolution</div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={scExample} margin={{top:5,right:10,left:0,bottom:0}}>
          <CartesianGrid strokeDasharray="3 3" stroke={BD}/>
          <XAxis dataKey="t" tick={{fill:SL,fontSize:9}} ticks={[0,60,120,180,240,300]} tickFormatter={v=>`16:${30+Math.floor(v/60)}`}/>
          <YAxis tick={{fill:SL,fontSize:9}} domain={['auto','auto']}/>
          <Tooltip content={<Tip/>}/>
          <Line type="monotone" dataKey="indPx" stroke={G} strokeWidth={2.5} dot={false} name="Indicative Price"/>
          {[{x:30,l:"Early opp.",c:"#22c55e"},{x:270,l:"Late opp.",c:"#22c55e"}].map((p,i)=>(
            <ReferenceLine key={i} x={p.x} stroke={p.c} strokeDasharray="4 3" label={{value:p.l,fill:p.c,fontSize:9,position:"top"}}/>
          ))}
        </LineChart>
      </ResponsiveContainer></div>
  </div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10}}>
    {[{t:"Must Complete",c:"#3b82f6",pct:"10%",d:"Guaranteed fill — tracks indicative size, always present in the book"},
      {t:"Target Quantity",c:"#f97316",pct:"10%",d:"Cancellable if cost surprise detected — provides fill buffer above minimum"},
      {t:"Opportunistic",c:"#22c55e",pct:"60%",d:"Deployed early to influence price formation OR late to capture liquidity at low cost"}
    ].map((q,i)=>(<div key={i} style={{background:C1,borderRadius:8,padding:10,border:`1px solid ${BD}`,borderLeft:`3px solid ${q.c}`}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:3}}>
        <span style={{fontSize:12,fontWeight:700,color:q.c}}>{q.t}</span>
        <span style={{fontSize:16,fontWeight:800,color:q.c}}>{q.pct}</span></div>
      <div style={{fontSize:10,color:SL,lineHeight:1.4}}>{q.d}</div></div>))}
  </div></div>);

const S11=()=>(<div style={{padding:"16px 28px"}}>
  <div style={{fontSize:12,color:SL,marginBottom:12}}>By setting aggressive limits, Smart Close can <strong style={{color:"#fff"}}>unlock excess volume beyond the target participation rate</strong>. The trade-off: more aggressive limits = more volume but higher price impact.</div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20,marginBottom:12}}>
    <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}>
      <div style={{fontSize:12,fontWeight:700,color:LB,marginBottom:2}}>Excess Volume Unlocked vs Limit Aggressiveness</div>
      <div style={{fontSize:10,color:"#475569",marginBottom:6}}>POV=25% | By time into auction (progressively more available)</div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={exVolData} margin={{top:5,right:10,left:0,bottom:0}}>
          <CartesianGrid strokeDasharray="3 3" stroke={BD}/>
          <XAxis dataKey="lim" tick={{fill:SL,fontSize:10}} label={{value:"Limit Aggressiveness (bps/vol)",position:"insideBottom",offset:-2,fill:"#64748b",fontSize:10}}/>
          <YAxis tick={{fill:SL,fontSize:10}} label={{value:"Excess Vol %",angle:-90,position:"insideLeft",fill:"#64748b",fontSize:10,dx:8}}/>
          <Tooltip content={<Tip fmt={v=>v+"%"}/>}/>
          <Line type="monotone" dataKey="vol1" stroke="#64748b" strokeWidth={1.5} dot={false} name="16:32"/>
          <Line type="monotone" dataKey="vol2" stroke="#f97316" strokeWidth={1.5} dot={false} name="16:33"/>
          <Line type="monotone" dataKey="vol3" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="16:34"/>
          <Line type="monotone" dataKey="vol4" stroke="#22c55e" strokeWidth={2.5} dot={false} name="16:35"/>
          <Legend wrapperStyle={{fontSize:10}}/>
        </LineChart>
      </ResponsiveContainer>
      <div style={{textAlign:"center",fontSize:10,color:SL}}>Later in auction → more excess volume available at same aggressiveness</div>
    </div>
    <div style={{background:C1,borderRadius:10,padding:"14px 14px 6px",border:`1px solid ${BD}`}}>
      <div style={{fontSize:12,fontWeight:700,color:LB,marginBottom:2}}>Price Impact vs Limit Aggressiveness</div>
      <div style={{fontSize:10,color:"#475569",marginBottom:6}}>POV=25% | Impact decreases later in auction (deeper book)</div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={exVolData} margin={{top:5,right:10,left:0,bottom:0}}>
          <CartesianGrid strokeDasharray="3 3" stroke={BD}/>
          <XAxis dataKey="lim" tick={{fill:SL,fontSize:10}} label={{value:"Limit Aggressiveness (bps/vol)",position:"insideBottom",offset:-2,fill:"#64748b",fontSize:10}}/>
          <YAxis tick={{fill:SL,fontSize:10}} label={{value:"Impact (bps/vol)",angle:-90,position:"insideLeft",fill:"#64748b",fontSize:10,dx:8}}/>
          <Tooltip content={<Tip fmt={v=>v+" bps/vol"}/>}/>
          <Line type="monotone" dataKey="impact1" stroke="#64748b" strokeWidth={1.5} dot={false} name="16:32"/>
          <Line type="monotone" dataKey="impact2" stroke="#f97316" strokeWidth={1.5} dot={false} name="16:33"/>
          <Line type="monotone" dataKey="impact3" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="16:34"/>
          <Line type="monotone" dataKey="impact4" stroke="#22c55e" strokeWidth={2.5} dot={false} name="16:35"/>
          <Legend wrapperStyle={{fontSize:10}}/>
        </LineChart>
      </ResponsiveContainer>
      <div style={{textAlign:"center",fontSize:10,color:SL}}>Later = lower impact per unit of aggressiveness (book deepens)</div>
    </div>
  </div>
  <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:16}}>
    <CO label="TRADE-OFF">The optimal limit aggressiveness balances excess volume capture against price impact. Smart Close calibrates this dynamically — more aggressive late in the auction when the book is deeper and impact is lower.</CO>
    <div style={{background:C1,borderRadius:10,padding:12,border:`1px solid ${BD}`}}>
      <div style={{fontSize:11,fontWeight:700,color:G,marginBottom:4}}>Real Example: BATS LN</div>
      <div style={{fontSize:10,color:SL,lineHeight:1.5}}>At POV=25% on 13 May 2025, Smart Close unlocked <strong style={{color:"#22c55e"}}>22% excess volume</strong> at 1.5× limit aggressiveness by timing the opportunistic slice to the final minute.</div>
    </div>
  </div><Src t="Source: RBC Capital Markets / Execution Data"/></div>);

const S12=()=>(<div style={{padding:"16px 28px"}}>
  <div style={{fontSize:12,color:SL,marginBottom:16}}>Every slide in this deck maps to a specific algo design decision. Here's the complete picture.</div>
  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
    <div>
      <div style={{fontSize:13,fontWeight:700,color:"#fff",marginBottom:10}}>Research → Design Mapping</div>
      {[{r:"Limit price becomes close at high POV",d:"Dynamic limit adjustment per participation",c:"#ef4444"},
        {r:"L3 transparency leaks information",d:"Auto-slicing across price levels",c:"#f97316"},
        {r:"Auction Angels shift imbalance in last 20s",d:"Real-time imbalance monitoring + offset logic",c:"#22c55e"},
        {r:"Static models underestimate cost",d:"Live CRS with place-review loop",c:"#3b82f6"},
        {r:"75% pairs early, 25% in final 30s",d:"Three-tier quantity: must-complete / target / opportunistic",c:G},
        {r:"Impact decreases later as book deepens",d:"Late opportunistic deployment for excess volume",c:"#a855f7"}
      ].map((r,i)=>(<div key={i} style={{display:"flex",gap:8,marginBottom:6,alignItems:"flex-start"}}>
        <div style={{minWidth:3,height:30,background:r.c,borderRadius:2,marginTop:2}}/>
        <div><div style={{fontSize:10,color:SL}}>{r.r}</div><div style={{fontSize:10,fontWeight:700,color:r.c}}>→ {r.d}</div></div></div>))}
    </div>
    <div style={{background:C1,borderRadius:10,padding:16,border:`1px solid ${BD}`}}>
      <div style={{fontSize:13,fontWeight:700,color:LB,marginBottom:14}}>Smart Close Differentiation</div>
      {[{t:"L3 Intelligence",d:"Only EMEA close algo with real-time L3 order book integration",c:"#ef4444"},
        {t:"Adaptive Impact Model",d:"Live cost surprise detection replaces static historical estimates",c:"#f97316"},
        {t:"Three-Tier Execution",d:"Must-complete / target / opportunistic framework matches urgency to market conditions",c:"#22c55e"},
        {t:"Excess Liquidity Capture",d:"Opportunistic component unlocks volume beyond target POV at calibrated cost",c:"#3b82f6"}
      ].map((f,i)=>(<div key={i} style={{marginBottom:10,paddingBottom:10,borderBottom:i<3?`1px solid ${BD}`:"none"}}>
        <div style={{fontSize:12,fontWeight:700,color:f.c,marginBottom:2}}>{f.t}</div>
        <div style={{fontSize:11,color:SL,lineHeight:1.4}}>{f.d}</div></div>))}
    </div>
  </div>
  <div style={{marginTop:12,background:`linear-gradient(90deg,rgba(196,160,74,0.12),rgba(0,51,102,0.2))`,border:`1px solid rgba(196,160,74,0.3)`,borderRadius:10,padding:"14px 20px",textAlign:"center"}}>
    <div style={{fontSize:14,fontWeight:700,color:"#fff"}}>From Microstructure Research to Production Algo</div>
    <div style={{fontSize:12,color:SL,marginTop:4}}>Every feature of Smart Close is grounded in the L3 evidence presented in this deck. We don't just trade the close — we understand it at the order level.</div>
  </div></div>);

const slides=[
  {c:S1,t:"How the Close Auction Works",s:"Mechanics"},
  {c:S2,t:"Call Phase Volume & Volatility",s:"Timing"},
  {c:S3,t:"The 6 Phases of the Close Auction",s:"Framework"},
  {c:S4,t:"Auction Angels — The Final 20 Seconds",s:"Price Formation"},
  {c:S5,t:"Three Key Challenges",s:"What Can Go Wrong"},
  {c:S6,t:"Orderbook Activities & Information Leakage",s:"Visibility Problem"},
  {c:S7,t:"Dynamic Impact Awareness",s:"Static Models Fail"},
  {c:S8,t:"Designing Smart Close",s:"Algo Architecture"},
  {c:S9,t:"Close Response System (CRS)",s:"Place-Review Loop"},
  {c:S10,t:"Smart Close in Action",s:"Three-Tier Example"},
  {c:S11,t:"Unlocking Excess Close Liquidity",s:"Volume vs Impact"},
  {c:S12,t:"Research → Algo Design",s:"Complete Picture"},
];

export default function Deck(){
  const[idx,setIdx]=useState(0);const S=slides[idx];const Comp=S.c;
  return(<div style={{background:BG,minHeight:"100vh",fontFamily:"'Segoe UI',system-ui,sans-serif",color:"#e2e8f0",paddingBottom:50}}>
    <SH num={idx+1} total={slides.length} title={S.t} subtitle={S.s}/>
    <Comp/>
    <div style={{position:"fixed",bottom:0,left:0,right:0,background:"rgba(15,23,42,0.95)",borderTop:`1px solid ${BD}`,padding:"8px 20px",display:"flex",justifyContent:"space-between",alignItems:"center",backdropFilter:"blur(10px)",zIndex:10}}>
      <button onClick={()=>setIdx(Math.max(0,idx-1))} disabled={idx===0} style={{padding:"5px 14px",borderRadius:6,border:`1px solid ${BD}`,background:idx===0?"transparent":C1,color:idx===0?"#475569":"#e2e8f0",fontSize:12,cursor:idx===0?"default":"pointer"}}>← Prev</button>
      <div style={{display:"flex",gap:3,flexWrap:"wrap",justifyContent:"center"}}>{slides.map((s,i)=>(<button key={i} onClick={()=>setIdx(i)} style={{width:i===idx?20:7,height:7,borderRadius:4,border:"none",background:i===idx?G:i<8?BD:"#6366f1",cursor:"pointer",transition:"all 0.2s",opacity:i===idx?1:.7}} title={s.t}/>))}</div>
      <button onClick={()=>setIdx(Math.min(slides.length-1,idx+1))} disabled={idx===slides.length-1} style={{padding:"5px 14px",borderRadius:6,border:`1px solid ${idx===slides.length-1?BD:G}`,background:idx===slides.length-1?"transparent":"rgba(196,160,74,0.15)",color:idx===slides.length-1?"#475569":G,fontSize:12,fontWeight:600,cursor:idx===slides.length-1?"default":"pointer"}}>Next →</button>
    </div>
  </div>);
}
