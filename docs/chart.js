/**
 * The accuracy-versus-pool-size chart, drawn from the saved ablation results.
 * Redraws on theme change so the series colours follow the page tokens.
 */
(function(){
  const DATA = {
    pools:[2,3,5,8,10,15,20,50,100,200,306],
    series:[
      {key:"dinov2", label:"DINOv2", slope:"−0.79 pts/doubling", cssvar:"--blue",
       v:[.9892,.9843,.9772,.9730,.9707,.9657,.9620,.9516,.9441,.9365,.9314],
       lo:[.9864,.9829,.9758,.9716,.9669,.9616,.9598,.9456,.9403,.9349,.9314],
       hi:[.9911,.9888,.9805,.9752,.9740,.9698,.9651,.9539,.9486,.9391,.9314]},
      {key:"clip", label:"CLIP", slope:"−2.06 pts/doubling", cssvar:"--red",
       v:[.9799,.9709,.9556,.9469,.9397,.9288,.9204,.8951,.8720,.8454,.8291],
       lo:[.9769,.9698,.9521,.9456,.9349,.9255,.9148,.8918,.8681,.8415,.8291],
       hi:[.9834,.9716,.9568,.9497,.9432,.9308,.9279,.8989,.8752,.8504,.8291]}
    ]
  };
  const W=720,H=400,M={t:18,r:64,b:52,l:52};
  const iw=W-M.l-M.r, ih=H-M.t-M.b;
  const yMin=81, yMax=100.4;
  const lx=Math.log2(2), lxMax=Math.log2(306);
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
    // Spans the whole axis: CLIP now reaches 82.9, so ticks stopping at 93 would
    // leave its entire curve in an unlabelled void.
    [82,84,86,88,90,92,94,96,98,100].forEach(t=>{
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
    '<tr><td>'+p+(p===306?" (full)":"")+'</td><td>'+(DATA.series[0].v[i]*100).toFixed(1)+
    '%</td><td>'+(DATA.series[1].v[i]*100).toFixed(1)+'%</td></tr>').join("");

  draw();
  const mq=window.matchMedia("(prefers-color-scheme: dark)");
  if(mq.addEventListener) mq.addEventListener("change",draw);
  new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
  window.addEventListener("resize",function(){tip.style.opacity="0";});
})();
