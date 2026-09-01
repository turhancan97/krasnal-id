/**
 * The accuracy-versus-pool-size chart, drawn from the saved ablation results.
 * Redraws on theme change so the series colours follow the page tokens.
 */
(function(){
  const DATA = {
    pools:[2,3,5,8,10,15,20,23],
    series:[
      {key:"dinov2", label:"DINOv2", slope:"−0.96 pts/doubling", cssvar:"--blue",
       v:[.9890,.9877,.9890,.9726,.9671,.9644,.9616,.9589],
       lo:[.9863,.9863,.9795,.9658,.9589,.9589,.9589,.9589],
       hi:[.9932,.9932,1.0000,.9795,.9726,.9658,.9658,.9589]},
      {key:"clip", label:"CLIP", slope:"−1.76 pts/doubling", cssvar:"--red",
       v:[.9890,.9753,.9767,.9589,.9575,.9397,.9315,.9247],
       lo:[.9795,.9726,.9658,.9452,.9452,.9315,.9247,.9247],
       hi:[1.0000,.9863,.9863,.9726,.9726,.9521,.9384,.9247]}
    ]
  };
  const W=720,H=400,M={t:18,r:64,b:52,l:52};
  const iw=W-M.l-M.r, ih=H-M.t-M.b;
  const yMin=92, yMax=100.4;
  const lx=Math.log2(2), lxMax=Math.log2(23);
  const X=p=>M.l+(Math.log2(p)-lx)/(lxMax-lx)*iw;
  const Y=v=>M.t+(yMax-v*100)/(yMax-yMin)*ih;
  const NS="http://www.w3.org/2000/svg";
  const el=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
  const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const svg=document.getElementById("chart");
  const tip=document.getElementById("tip");

  function wire(colors){
    const hit=document.getElementById("hit-area"), ch=document.getElementById("ch");
    function near(evt){
      const r=svg.getBoundingClientRect();
      const sx=(evt.clientX-r.left)/r.width*W;
      let best=0,bd=1e9;
      DATA.pools.forEach((p,i)=>{const d=Math.abs(X(p)-sx); if(d<bd){bd=d;best=i;}});
      return best;
    }
    function show(evt){
      const i=near(evt), p=DATA.pools[i];
      ch.setAttribute("x1",X(p)); ch.setAttribute("x2",X(p)); ch.style.opacity=".55";
      tip.innerHTML='<div class="hd">pool of '+p+'</div>'+DATA.series.map((s,si)=>
        '<div class="row"><span class="sw" style="background:'+colors[si]+'"></span>'+
        s.label+' '+(s.v[i]*100).toFixed(1)+'%</div>').join("");
      const r=svg.getBoundingClientRect();
      tip.style.left=(X(p)/W*r.width)+"px";
      tip.style.top=(Y(Math.max(DATA.series[0].v[i],DATA.series[1].v[i]))/H*r.height)+"px";
      tip.style.opacity="1";
    }
    function hide(){ch.style.opacity="0"; tip.style.opacity="0";}
    hit.addEventListener("pointermove",show);
    hit.addEventListener("pointerleave",hide);
    hit.addEventListener("pointerdown",show);
  }

  function draw(){
    svg.textContent="";
    const colors=DATA.series.map(s=>css(s.cssvar));
    [93,94,95,96,97,98,99,100].forEach(t=>{
      svg.appendChild(el("line",{class:"grid",x1:M.l,x2:M.l+iw,y1:Y(t/100),y2:Y(t/100)}));
      const tx=el("text",{x:M.l-10,y:Y(t/100)+4,"text-anchor":"end"});
      tx.textContent=t; svg.appendChild(tx);
    });
    svg.appendChild(el("line",{class:"axis",x1:M.l,x2:M.l+iw,y1:M.t+ih,y2:M.t+ih}));
    DATA.pools.forEach(p=>{
      const tx=el("text",{x:X(p),y:M.t+ih+20,"text-anchor":"middle"});
      tx.textContent=p; svg.appendChild(tx);
    });
    const ax=el("text",{class:"axis-title",x:M.l+iw/2,y:M.t+ih+43,"text-anchor":"middle"});
    ax.textContent="candidate pool size"; svg.appendChild(ax);
    const ay=el("text",{class:"axis-title",x:14,y:M.t+ih/2,"text-anchor":"middle",
      transform:"rotate(-90 14 "+(M.t+ih/2)+")"});
    ay.textContent="top-1 accuracy %"; svg.appendChild(ay);

    DATA.series.forEach((s,si)=>{
      const up=DATA.pools.map((p,i)=>X(p)+","+Y(s.hi[i]));
      const dn=DATA.pools.map((p,i)=>X(p)+","+Y(s.lo[i])).reverse();
      svg.appendChild(el("polygon",{points:up.concat(dn).join(" "),
        fill:colors[si],"fill-opacity":".14",stroke:"none"}));
    });
    DATA.series.forEach((s,si)=>{
      svg.appendChild(el("polyline",{class:"line",stroke:colors[si],
        points:DATA.pools.map((p,i)=>X(p)+","+Y(s.v[i])).join(" ")}));
      DATA.pools.forEach((p,i)=>{
        svg.appendChild(el("circle",{class:"dot",cx:X(p),cy:Y(s.v[i]),r:4,fill:colors[si],
          stroke:css("--surface")}));
      });
      const last=DATA.pools.length-1;
      const lab=el("text",{class:"dlabel",x:X(DATA.pools[last])+12,y:Y(s.v[last])+4,fill:colors[si]});
      lab.textContent=s.label; svg.appendChild(lab);
    });
    svg.appendChild(el("line",{class:"crosshair",id:"ch",y1:M.t,y2:M.t+ih,x1:0,x2:0}));
    svg.appendChild(el("rect",{class:"hit-area",id:"hit",x:M.l,y:M.t,width:iw,height:ih}));
    wire(colors);
  }

  document.getElementById("legend").innerHTML=DATA.series.map(s=>
    '<span class="k"><span class="sw" style="background:var('+s.cssvar+')"></span>'+
    s.label+' <span class="slope">'+s.slope+'</span></span>').join("");
  document.getElementById("tbody").innerHTML=DATA.pools.map((p,i)=>
    '<tr><td>'+p+(p===23?" (full)":"")+'</td><td>'+(DATA.series[0].v[i]*100).toFixed(1)+
    '%</td><td>'+(DATA.series[1].v[i]*100).toFixed(1)+'%</td></tr>').join("");

  draw();
  const mq=window.matchMedia("(prefers-color-scheme: dark)");
  if(mq.addEventListener) mq.addEventListener("change",draw);
  new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
  window.addEventListener("resize",function(){tip.style.opacity="0";});
})();
