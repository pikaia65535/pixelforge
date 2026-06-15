import React, { useState, useRef, useEffect, useCallback } from "react";
import ReactDOM from "react-dom/client";

const C = {
  bg:"#0a0a0c", surface:"#111114", surface2:"#18181d", surface3:"#1f1f26",
  border:"#272730", text:"#eeeef5", muted:"#5a5a70",
  accent:"#7c6dfa", accent2:"#fa6d8a", green:"#4ade9a", yellow:"#f5c542", red:"#f46d6d",
};

// ── 보간법 ────────────────────────────────────
function cubicWeight(t){const a=-0.5;t=Math.abs(t);if(t<=1)return(a+2)*t*t*t-(a+3)*t*t+1;if(t<=2)return a*t*t*t-5*a*t*t+8*a*t-4*a;return 0;}
function bicubicResize(sd,sw,sh,dw,dh){
  const dst=new Uint8ClampedArray(dw*dh*4),xR=sw/dw,yR=sh/dh;
  for(let dy=0;dy<dh;dy++)for(let dx=0;dx<dw;dx++){
    const sx=dx*xR,sy=dy*yR,x0=Math.floor(sx),y0=Math.floor(sy);
    let r=0,g=0,b=0,a=0,wS=0;
    for(let m=-1;m<=2;m++)for(let n=-1;n<=2;n++){
      const px=Math.min(Math.max(x0+n,0),sw-1),py=Math.min(Math.max(y0+m,0),sh-1);
      const w=cubicWeight(sx-(x0+n))*cubicWeight(sy-(y0+m)),i=(py*sw+px)*4;
      r+=sd[i]*w;g+=sd[i+1]*w;b+=sd[i+2]*w;a+=sd[i+3]*w;wS+=w;
    }
    const di=(dy*dw+dx)*4;
    dst[di]=Math.min(255,Math.max(0,Math.round(r/wS)));dst[di+1]=Math.min(255,Math.max(0,Math.round(g/wS)));
    dst[di+2]=Math.min(255,Math.max(0,Math.round(b/wS)));dst[di+3]=Math.min(255,Math.max(0,Math.round(a/wS)));
  }
  return dst;
}
function bilinearResize(sd,sw,sh,dw,dh){
  const dst=new Uint8ClampedArray(dw*dh*4),xR=sw/dw,yR=sh/dh;
  for(let dy=0;dy<dh;dy++)for(let dx=0;dx<dw;dx++){
    const sx=dx*xR,sy=dy*yR,x0=Math.floor(sx),y0=Math.floor(sy);
    const x1=Math.min(x0+1,sw-1),y1=Math.min(y0+1,sh-1),fx=sx-x0,fy=sy-y0;
    const i00=(y0*sw+x0)*4,i10=(y0*sw+x1)*4,i01=(y1*sw+x0)*4,i11=(y1*sw+x1)*4,di=(dy*dw+dx)*4;
    for(let c=0;c<4;c++)dst[di+c]=Math.round(sd[i00+c]*(1-fx)*(1-fy)+sd[i10+c]*fx*(1-fy)+sd[i01+c]*(1-fx)*fy+sd[i11+c]*fx*fy);
  }
  return dst;
}
function nearestResize(sd,sw,sh,dw,dh){
  const dst=new Uint8ClampedArray(dw*dh*4),xR=sw/dw,yR=sh/dh;
  for(let dy=0;dy<dh;dy++)for(let dx=0;dx<dw;dx++){
    const sx=Math.min(Math.floor(dx*xR),sw-1),sy=Math.min(Math.floor(dy*yR),sh-1);
    const si=(sy*sw+sx)*4,di=(dy*dw+dx)*4;
    dst[di]=sd[si];dst[di+1]=sd[si+1];dst[di+2]=sd[si+2];dst[di+3]=sd[si+3];
  }
  return dst;
}
function boxBlurPass(src,dst,w,h,r,horiz){
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    let r2=0,g=0,b=0,a=0,cnt=0;
    for(let i=-r;i<=r;i++){
      const px=horiz?Math.min(Math.max(x+i,0),w-1):x,py=horiz?y:Math.min(Math.max(y+i,0),h-1),idx=(py*w+px)*4;
      r2+=src[idx];g+=src[idx+1];b+=src[idx+2];a+=src[idx+3];cnt++;
    }
    const di=(y*w+x)*4;dst[di]=r2/cnt;dst[di+1]=g/cnt;dst[di+2]=b/cnt;dst[di+3]=a/cnt;
  }
}
function gaussianBlurRegion(imgData,rx,ry,rw,rh,cw,ch,radius){
  const r=Math.max(1,Math.round(radius)),src=new Float32Array(imgData.data),tmp=new Float32Array(src.length),out=new Float32Array(src.length);
  boxBlurPass(src,tmp,cw,ch,r,true);boxBlurPass(tmp,out,cw,ch,r,false);
  boxBlurPass(out,tmp,cw,ch,r,true);boxBlurPass(tmp,out,cw,ch,r,false);
  boxBlurPass(out,tmp,cw,ch,r,true);boxBlurPass(tmp,out,cw,ch,r,false);
  for(let row=ry;row<Math.min(ry+rh,ch);row++)for(let col=rx;col<Math.min(rx+rw,cw);col++){
    const i=(row*cw+col)*4;imgData.data[i]=out[i];imgData.data[i+1]=out[i+1];imgData.data[i+2]=out[i+2];imgData.data[i+3]=out[i+3];
  }
}
async function runPipeline(imgEl,steps){
  const canvas=document.createElement("canvas");
  canvas.width=imgEl.width;canvas.height=imgEl.height;
  let ctx=canvas.getContext("2d");
  ctx.drawImage(imgEl,0,0);
  for(const step of steps){
    if(step.type==="resize"){
      const sw=canvas.width,sh=canvas.height;
      let dw=sw,dh=sh;
      if(step.scaleMode==="ratio"){dw=Math.max(1,Math.round(sw*step.ratio));dh=Math.max(1,Math.round(sh*step.ratio));}
      else{dw=step.outW||sw;dh=step.outH||sh;}
      const sd=ctx.getImageData(0,0,sw,sh).data;
      let pd;
      if(step.method==="nearest")pd=nearestResize(sd,sw,sh,dw,dh);
      else if(step.method==="bilinear")pd=bilinearResize(sd,sw,sh,dw,dh);
      else pd=bicubicResize(sd,sw,sh,dw,dh);
      canvas.width=dw;canvas.height=dh;ctx=canvas.getContext("2d");
      ctx.putImageData(new ImageData(new Uint8ClampedArray(pd),dw,dh),0,0);
    } else if(step.type==="crop"){
      const cx=Math.round(step.x),cy=Math.round(step.y),cw=Math.round(step.w),ch=Math.round(step.h);
      if(cw>0&&ch>0){
        const tmp=document.createElement("canvas");tmp.width=cw;tmp.height=ch;
        tmp.getContext("2d").drawImage(canvas,cx,cy,cw,ch,0,0,cw,ch);
        canvas.width=cw;canvas.height=ch;ctx=canvas.getContext("2d");ctx.drawImage(tmp,0,0);
      }
    } else if(step.type==="rotate"){
      const w=canvas.width,h=canvas.height,rad=step.deg*Math.PI/180;
      const sin=Math.abs(Math.sin(rad)),cos=Math.abs(Math.cos(rad));
      const nw=Math.round(w*cos+h*sin),nh=Math.round(w*sin+h*cos);
      const tmp=document.createElement("canvas");tmp.width=nw;tmp.height=nh;
      const tc=tmp.getContext("2d");tc.save();tc.translate(nw/2,nh/2);tc.rotate(rad);
      if(step.flipH)tc.scale(-1,1);if(step.flipV)tc.scale(1,-1);
      tc.drawImage(canvas,-w/2,-h/2);tc.restore();
      canvas.width=nw;canvas.height=nh;ctx=canvas.getContext("2d");ctx.drawImage(tmp,0,0);
    } else if(step.type==="watermark"){
      if(step.wmImg&&step.wmImg.el){
        const ww=Math.round(canvas.width*step.scale),wh=Math.round(ww*step.wmImg.h/step.wmImg.w),pad=20;
        const pm={topLeft:{x:pad,y:pad},topRight:{x:canvas.width-ww-pad,y:pad},center:{x:(canvas.width-ww)/2,y:(canvas.height-wh)/2},bottomLeft:{x:pad,y:canvas.height-wh-pad},bottomRight:{x:canvas.width-ww-pad,y:canvas.height-wh-pad}};
        const pos=pm[step.pos]||pm.bottomRight;
        ctx.globalAlpha=step.opacity;ctx.drawImage(step.wmImg.el,pos.x,pos.y,ww,wh);ctx.globalAlpha=1;
      }
    } else if(step.type==="blur"){
      if((step.rects||[]).length>0){
        const id=ctx.getImageData(0,0,canvas.width,canvas.height);
        for(const rect of step.rects)gaussianBlurRegion(id,rect.x,rect.y,rect.w,rect.h,canvas.width,canvas.height,step.blurRadius);
        ctx.putImageData(id,0,0);
      }
    }
  }
  return canvas;
}

function fmtBytes(b){if(b<1024)return b+" B";if(b<1048576)return(b/1024).toFixed(1)+" KB";return(b/1048576).toFixed(2)+" MB";}
let _uid=0;function nextId(){return ++_uid;}

const STEP_META={
  crop:     {icon:"✂️",label:"자르기",    color:"#60c8fa"},
  resize:   {icon:"📐",label:"리사이즈",  color:"#7c6dfa"},
  rotate:   {icon:"🔄",label:"회전/반전", color:"#4ade9a"},
  watermark:{icon:"💧",label:"워터마크",  color:"#f5c542"},
  blur:     {icon:"👤",label:"얼굴 흐리기",color:"#fa6d8a"},
};

function defaultStep(type,img){
  if(type==="crop")return{type,id:nextId(),x:0,y:0,w:img?img.w:100,h:img?img.h:100,aspect:"free"};
  if(type==="resize")return{type,id:nextId(),method:"bicubic",scaleMode:"ratio",ratio:2,outW:img?img.w*2:100,outH:img?img.h*2:100,lockAR:true};
  if(type==="rotate")return{type,id:nextId(),deg:90,flipH:false,flipV:false};
  if(type==="watermark")return{type,id:nextId(),wmImg:null,opacity:0.5,scale:0.3,pos:"bottomRight"};
  if(type==="blur")return{type,id:nextId(),blurRadius:20,rects:[]};
  return{type,id:nextId()};
}

const globalStyle=`
  *{box-sizing:border-box;margin:0;padding:0;}
  html,body,#root{height:100%;}
  body{background:#0a0a0c;overflow-x:hidden;}
  ::-webkit-scrollbar{width:3px;}::-webkit-scrollbar-thumb{background:#272730;border-radius:2px;}
  @keyframes spin{to{transform:rotate(360deg);}}
  @keyframes fadeIn{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}
  input[type=range]{width:100%;cursor:pointer;}
  .step-btn:hover{border-color:var(--hc)!important;background:var(--hbg)!important;}
  @media(max-width:768px){
    .editor-layout{flex-direction:column!important;}
    .sidebar{width:100%!important;max-height:50vh;overflow-y:auto;}
    .preview-area{min-height:40vh!important;}
  }
`;

// ── UI 부품 ───────────────────────────────────
function Btn({children,onClick,active,disabled,color,small,full}){
  const ac=color||C.accent;
  return React.createElement("button",{onClick,disabled,style:{padding:small?"0.22rem 0.5rem":"0.35rem 0.7rem",borderRadius:7,border:`1px solid ${active?ac:C.border}`,background:active?ac:C.surface3,color:active?"#fff":C.muted,fontSize:small?"0.7rem":"0.77rem",fontWeight:600,cursor:disabled?"not-allowed":"pointer",opacity:disabled?0.4:1,transition:"all 0.15s",fontFamily:"inherit",width:full?"100%":"auto",whiteSpace:"nowrap"}},children);
}
function Label({children}){return React.createElement("div",{style:{fontSize:"0.6rem",fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase",color:C.muted,marginBottom:"0.3rem"}},children);}
function Divider(){return React.createElement("hr",{style:{border:"none",borderTop:`1px solid ${C.border}`,margin:"0.65rem 0"}});}
function RangeInput({value,onChange,min,max,step,color}){return React.createElement("input",{type:"range",value,onChange,min,max,step:step||1,style:{accentColor:color||C.accent,width:"100%",cursor:"pointer"}});}
function NInput({value,onChange,min,max}){return React.createElement("input",{type:"number",value,onChange,min,max,style:{background:C.surface2,border:`1px solid ${C.border}`,borderRadius:6,padding:"0.28rem 0.45rem",color:C.text,fontSize:"0.77rem",outline:"none",fontFamily:"inherit",width:"100%"}});}

// ── 자르기 패널 ───────────────────────────────
function CropPanel({step,img,onChange}){
  const ref=useRef();
  const [drawing,setDrawing]=useState(false);
  const [start,setStart]=useState(null);
  const ac=STEP_META.crop.color;
  function upd(p){onChange({...step,...p});}
  function getIR(){
    const cv=ref.current;if(!cv||!img)return null;
    const rb=cv.getBoundingClientRect(),iR=img.w/img.h,cR=rb.width/rb.height;
    let dW,dH,oX,oY;
    if(iR>cR){dW=rb.width;dH=rb.width/iR;oX=0;oY=(rb.height-dH)/2;}
    else{dH=rb.height;dW=rb.height*iR;oX=(rb.width-dW)/2;oY=0;}
    return{rb,dW,dH,oX,oY};
  }
  function getPos(e){
    const ir=getIR();if(!ir)return null;
    const{rb,dW,dH,oX,oY}=ir;
    const clientX=e.touches?e.touches[0].clientX:e.clientX;
    const clientY=e.touches?e.touches[0].clientY:e.clientY;
    return{x:Math.round(Math.min(Math.max(clientX-rb.left-oX,0),dW)*(img.w/dW)),y:Math.round(Math.min(Math.max(clientY-rb.top-oY,0),dH)*(img.h/dH))};
  }
  function onStart(e){e.preventDefault();const p=getPos(e);if(!p)return;setDrawing(true);setStart(p);}
  function onMove(e){
    e.preventDefault();if(!drawing||!start)return;
    const p=getPos(e);if(!p)return;
    let x=Math.min(start.x,p.x),y=Math.min(start.y,p.y),w=Math.abs(p.x-start.x),h=Math.abs(p.y-start.y);
    if(step.aspect==="1:1"){const s=Math.min(w,h);w=s;h=s;}
    else if(step.aspect==="16:9")h=Math.round(w*9/16);
    else if(step.aspect==="4:3")h=Math.round(w*3/4);
    else if(step.aspect==="3:2")h=Math.round(w*2/3);
    upd({x,y,w,h});
  }
  function onEnd(){setDrawing(false);setStart(null);}
  function toScreen(){
    const ir=getIR();if(!ir)return null;
    const{dW,dH,oX,oY}=ir;
    return{left:oX+step.x*(dW/img.w),top:oY+step.y*(dH/img.h),width:step.w*(dW/img.w),height:step.h*(dH/img.h)};
  }
  const sc=toScreen();
  return React.createElement(React.Fragment,null,
    React.createElement(Label,null,"비율"),
    React.createElement("div",{style:{display:"flex",gap:3,flexWrap:"wrap",marginBottom:"0.6rem"}},
      ["free","1:1","16:9","4:3","3:2"].map(a=>React.createElement(Btn,{key:a,small:true,active:step.aspect===a,color:ac,onClick:()=>upd({aspect:a})},a))
    ),
    React.createElement(Label,null,"드래그로 영역 선택"),
    React.createElement("div",{style:{position:"relative",borderRadius:7,overflow:"hidden",background:`repeating-conic-gradient(${C.surface3} 0% 25%,${C.surface2} 0% 50%) 0 0/8px 8px`,marginBottom:"0.6rem",touchAction:"none",height:120}},
      React.createElement("img",{src:img.url,style:{width:"100%",height:"100%",objectFit:"contain",display:"block"}}),
      React.createElement("div",{ref,style:{position:"absolute",inset:0,cursor:"crosshair"},
        onMouseDown:onStart,onMouseMove:onMove,onMouseUp:onEnd,onMouseLeave:onEnd,
        onTouchStart:onStart,onTouchMove:onMove,onTouchEnd:onEnd}),
      sc&&step.w>0&&React.createElement("div",{style:{position:"absolute",left:sc.left,top:sc.top,width:sc.width,height:sc.height,border:`2px solid ${ac}`,background:`${ac}18`,pointerEvents:"none",borderRadius:2}})
    ),
    React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4}},
      React.createElement("div",null,React.createElement(Label,null,"X"),React.createElement(NInput,{value:step.x,min:0,max:img.w,onChange:e=>upd({x:parseInt(e.target.value)||0})})),
      React.createElement("div",null,React.createElement(Label,null,"Y"),React.createElement(NInput,{value:step.y,min:0,max:img.h,onChange:e=>upd({y:parseInt(e.target.value)||0})})),
      React.createElement("div",null,React.createElement(Label,null,"너비"),React.createElement(NInput,{value:step.w,min:1,max:img.w,onChange:e=>upd({w:parseInt(e.target.value)||1})})),
      React.createElement("div",null,React.createElement(Label,null,"높이"),React.createElement(NInput,{value:step.h,min:1,max:img.h,onChange:e=>upd({h:parseInt(e.target.value)||1})}))
    ),
    step.w>0&&React.createElement("div",{style:{marginTop:"0.5rem",background:C.surface2,borderRadius:6,padding:"0.35rem 0.6rem",fontSize:"0.68rem",color:C.muted}},
      React.createElement("span",{style:{color:C.text,fontWeight:600}},`${Math.round(step.w)}×${Math.round(step.h)}px`)," 선택됨"
    )
  );
}

// ── 단계 패널 ─────────────────────────────────
function StepPanel({step,img,onChange,onRemove,index,totalSteps,onMove}){
  const wmRef=useRef();
  const blurRef=useRef();
  const [drawing,setDrawing]=useState(false);
  const [drawStart,setDrawStart]=useState(null);
  const [drawRect,setDrawRect]=useState(null);
  const [blurMode,setBlurMode]=useState("manual");
  const [faceDetecting,setFaceDetecting]=useState(false);
  const [faceApiLoaded,setFaceApiLoaded]=useState(false);
  const [open,setOpen]=useState(true);
  const m=STEP_META[step.type];
  const ac=m?m.color:C.accent;
  function upd(p){onChange({...step,...p});}

  function getIR(){
    const cv=blurRef.current;if(!cv||!img)return null;
    const rb=cv.getBoundingClientRect(),iR=img.w/img.h,cR=rb.width/rb.height;
    let dW,dH,oX,oY;
    if(iR>cR){dW=rb.width;dH=rb.width/iR;oX=0;oY=(rb.height-dH)/2;}
    else{dH=rb.height;dW=rb.height*iR;oX=(rb.width-dW)/2;oY=0;}
    return{rb,dW,dH,oX,oY};
  }
  function getBlurPos(e){
    const ir=getIR();if(!ir)return null;
    const{rb,dW,dH,oX,oY}=ir;
    const clientX=e.touches?e.touches[0].clientX:e.clientX;
    const clientY=e.touches?e.touches[0].clientY:e.clientY;
    return{x:Math.round(Math.min(Math.max(clientX-rb.left-oX,0),dW)*(img.w/dW)),y:Math.round(Math.min(Math.max(clientY-rb.top-oY,0),dH)*(img.h/dH))};
  }
  function rToScreen(rect){
    const ir=getIR();if(!ir)return null;
    const{dW,dH,oX,oY}=ir;
    return{left:oX+rect.x*(dW/img.w),top:oY+rect.y*(dH/img.h),width:rect.w*(dW/img.w),height:rect.h*(dH/img.h)};
  }
  function onMD(e){e.preventDefault();const p=getBlurPos(e);if(!p)return;setDrawing(true);setDrawStart(p);setDrawRect(null);}
  function onMM(e){e.preventDefault();if(!drawing||!drawStart)return;const p=getBlurPos(e);if(!p)return;setDrawRect({x:Math.min(drawStart.x,p.x),y:Math.min(drawStart.y,p.y),w:Math.abs(p.x-drawStart.x),h:Math.abs(p.y-drawStart.y)});}
  function onMU(){if(!drawing)return;setDrawing(false);if(drawRect&&drawRect.w>5&&drawRect.h>5)upd({rects:[...(step.rects||[]),drawRect]});setDrawRect(null);setDrawStart(null);}
  async function detectFaces(){
    if(!img)return;setFaceDetecting(true);
    if(!faceApiLoaded){
      try{
        await new Promise((res,rej)=>{const s=document.createElement("script");s.src="https://cdnjs.cloudflare.com/ajax/libs/face-api.js/0.22.2/face-api.min.js";s.onload=res;s.onerror=rej;document.head.appendChild(s);});
        await window.faceapi.nets.tinyFaceDetector.loadFromUri("https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights");
        setFaceApiLoaded(true);
      }catch{alert("모델 로드 실패.");setFaceDetecting(false);return;}
    }
    try{
      const dets=await window.faceapi.detectAllFaces(img.el,new window.faceapi.TinyFaceDetectorOptions());
      if(!dets.length)alert("얼굴을 감지하지 못했어요.");
      else upd({rects:[...(step.rects||[]),...dets.map(d=>({x:Math.round(d.box.x),y:Math.round(d.box.y),w:Math.round(d.box.width),h:Math.round(d.box.height)}))]});
    }catch{alert("오류가 발생했어요.");}
    setFaceDetecting(false);
  }

  const allRects=[...(step.rects||[]),...(drawRect?[drawRect]:[])];

  let body=null;
  if(!open)body=null;
  else if(step.type==="crop")body=React.createElement(CropPanel,{step,img,onChange});
  else if(step.type==="resize")body=React.createElement(React.Fragment,null,
    React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4,marginBottom:"0.6rem"}},
      React.createElement(Btn,{active:step.scaleMode==="ratio",color:ac,onClick:()=>upd({scaleMode:"ratio"}),small:true},"배율"),
      React.createElement(Btn,{active:step.scaleMode==="px",color:ac,onClick:()=>upd({scaleMode:"px"}),small:true},"직접(px)")
    ),
    step.scaleMode==="ratio"
      ?React.createElement(React.Fragment,null,
          React.createElement(Label,null,`배율 ×${Number(step.ratio).toFixed(1)}`),
          React.createElement(RangeInput,{value:step.ratio,min:"0.1",max:"8",step:"0.1",color:ac,onChange:e=>upd({ratio:parseFloat(e.target.value)})}),
          React.createElement("div",{style:{display:"flex",gap:3,marginTop:"0.4rem",flexWrap:"wrap"}},
            [0.25,0.5,1,2,3,4].map(r=>React.createElement(Btn,{key:r,small:true,active:step.ratio===r,color:ac,onClick:()=>upd({ratio:r})},`×${r}`))
          )
        )
      :React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr auto 1fr",gap:4,alignItems:"end"}},
          React.createElement("div",null,React.createElement(Label,null,"너비"),React.createElement(NInput,{value:step.outW||"",min:1,max:8000,onChange:e=>{const w=parseInt(e.target.value)||1;upd({outW:w,outH:step.lockAR&&img?Math.round(w*img.h/img.w):step.outH});}})),
          React.createElement("button",{onClick:()=>upd({lockAR:!step.lockAR}),style:{background:step.lockAR?ac:C.surface3,border:`1px solid ${step.lockAR?ac:C.border}`,borderRadius:5,padding:"0.28rem",cursor:"pointer",fontSize:"0.75rem",marginBottom:0}},step.lockAR?"🔒":"🔓"),
          React.createElement("div",null,React.createElement(Label,null,"높이"),React.createElement(NInput,{value:step.outH||"",min:1,max:8000,onChange:e=>{const h=parseInt(e.target.value)||1;upd({outH:h,outW:step.lockAR&&img?Math.round(h*img.w/img.h):step.outW});}}))
        ),
    React.createElement(Divider,null),
    React.createElement(Label,null,"보간법"),
    React.createElement("div",{style:{display:"flex",flexDirection:"column",gap:3}},
      [{id:"nearest",l:"Nearest",d:"픽셀아트"},{id:"bilinear",l:"Bilinear",d:"4픽셀 혼합"},{id:"bicubic",l:"Bicubic ✦",d:"최고품질"}].map(mm=>
        React.createElement("div",{key:mm.id,onClick:()=>upd({method:mm.id}),style:{padding:"0.35rem 0.6rem",borderRadius:6,cursor:"pointer",border:`1px solid ${step.method===mm.id?ac:C.border}`,background:step.method===mm.id?`${ac}18`:C.surface2,display:"flex",justifyContent:"space-between",alignItems:"center"}},
          React.createElement("span",{style:{fontWeight:600,fontSize:"0.73rem",color:step.method===mm.id?ac:C.text}},mm.l),
          React.createElement("span",{style:{fontSize:"0.62rem",color:C.muted}},mm.d)
        )
      )
    )
  );
  else if(step.type==="rotate")body=React.createElement(React.Fragment,null,
    React.createElement(Label,null,`회전 ${step.deg}°`),
    React.createElement(RangeInput,{value:step.deg,min:0,max:359,color:ac,onChange:e=>upd({deg:parseInt(e.target.value)})}),
    React.createElement("div",{style:{display:"flex",gap:3,marginTop:"0.4rem",flexWrap:"wrap"}},
      [0,90,180,270].map(d=>React.createElement(Btn,{key:d,small:true,active:step.deg===d,color:ac,onClick:()=>upd({deg:d})},`${d}°`))
    ),
    React.createElement(Divider,null),
    React.createElement(Label,null,"반전"),
    React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4}},
      React.createElement(Btn,{active:step.flipH,color:ac,onClick:()=>upd({flipH:!step.flipH}),small:true},"↔ 좌우"),
      React.createElement(Btn,{active:step.flipV,color:ac,onClick:()=>upd({flipV:!step.flipV}),small:true},"↕ 상하")
    )
  );
  else if(step.type==="watermark")body=React.createElement(React.Fragment,null,
    React.createElement("div",{onClick:()=>wmRef.current.click(),style:{border:`2px dashed ${step.wmImg?ac:C.border}`,borderRadius:7,padding:"0.55rem",textAlign:"center",cursor:"pointer",marginBottom:"0.6rem",background:C.surface2}},
      step.wmImg
        ?React.createElement("div",{style:{display:"flex",alignItems:"center",gap:6}},React.createElement("img",{src:step.wmImg.url,style:{width:28,height:28,objectFit:"contain",borderRadius:4}}),React.createElement("span",{style:{fontSize:"0.72rem",color:C.green,fontWeight:600}},"✓ 로드됨"))
        :React.createElement("span",{style:{fontSize:"0.73rem",color:C.muted}},"탭해서 이미지 선택")
    ),
    React.createElement("input",{ref:wmRef,type:"file",accept:"image/*",style:{display:"none"},onChange:e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=ev=>{const url=ev.target.result;const im=new Image();im.onload=()=>upd({wmImg:{url,w:im.width,h:im.height,el:im}});im.src=url;};r.readAsDataURL(f);}}),
    React.createElement(Label,null,`투명도 ${Math.round(step.opacity*100)}%`),
    React.createElement(RangeInput,{value:step.opacity,min:0.05,max:1,step:0.05,color:ac,onChange:e=>upd({opacity:parseFloat(e.target.value)})}),
    React.createElement(Divider,null),
    React.createElement(Label,null,`크기 ${Math.round(step.scale*100)}%`),
    React.createElement(RangeInput,{value:step.scale,min:0.05,max:0.8,step:0.05,color:ac,onChange:e=>upd({scale:parseFloat(e.target.value)})}),
    React.createElement(Divider,null),
    React.createElement(Label,null,"위치"),
    React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:3}},
      [{id:"topLeft",l:"↖ 좌상"},{id:"topRight",l:"↗ 우상"},{id:"center",l:"⊕ 중앙"},{id:"bottomLeft",l:"↙ 좌하"},{id:"bottomRight",l:"↘ 우하"}].map(p=>
        React.createElement(Btn,{key:p.id,small:true,active:step.pos===p.id,color:ac,onClick:()=>upd({pos:p.id})},p.l)
      )
    )
  );
  else if(step.type==="blur")body=React.createElement(React.Fragment,null,
    React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4,marginBottom:"0.55rem"}},
      React.createElement(Btn,{active:blurMode==="auto",color:ac,onClick:()=>setBlurMode("auto"),small:true},"🤖 자동"),
      React.createElement(Btn,{active:blurMode==="manual",color:ac,onClick:()=>setBlurMode("manual"),small:true},"✏️ 수동")
    ),
    blurMode==="auto"&&React.createElement("button",{onClick:detectFaces,disabled:faceDetecting,style:{width:"100%",padding:"0.42rem",background:faceDetecting?C.surface3:ac,border:"none",borderRadius:7,color:faceDetecting?C.muted:"#fff",fontWeight:700,fontSize:"0.74rem",cursor:"pointer",marginBottom:"0.55rem",fontFamily:"inherit"}},faceDetecting?"모델 로딩 중...":"🔍 자동 감지"),
    blurMode==="manual"&&img&&React.createElement("div",{style:{position:"relative",marginBottom:"0.55rem",borderRadius:7,overflow:"hidden",background:`repeating-conic-gradient(${C.surface3} 0% 25%,${C.surface2} 0% 50%) 0 0/8px 8px`,touchAction:"none",height:110}},
      React.createElement("img",{src:img.url,style:{width:"100%",height:"100%",objectFit:"contain",display:"block"}}),
      React.createElement("div",{ref:blurRef,style:{position:"absolute",inset:0,cursor:"crosshair"},
        onMouseDown:onMD,onMouseMove:onMM,onMouseUp:onMU,onMouseLeave:onMU,
        onTouchStart:onMD,onTouchMove:onMM,onTouchEnd:onMU}),
      allRects.map((rect,i)=>{const s=rToScreen(rect);if(!s)return null;return React.createElement("div",{key:i,style:{position:"absolute",left:s.left,top:s.top,width:s.width,height:s.height,border:`2px solid ${ac}`,background:`${ac}22`,borderRadius:2,pointerEvents:"none"}});})
    ),
    React.createElement(Label,null,`블러 강도 ${step.blurRadius}`),
    React.createElement(RangeInput,{value:step.blurRadius,min:5,max:50,color:ac,onChange:e=>upd({blurRadius:parseInt(e.target.value)})}),
    React.createElement("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginTop:"0.4rem"}},
      React.createElement("span",{style:{fontSize:"0.67rem",color:C.muted}},`${(step.rects||[]).length}개 선택`),
      (step.rects||[]).length>0&&React.createElement("button",{onClick:()=>upd({rects:[]}),style:{background:"none",border:`1px solid ${C.border}`,borderRadius:5,padding:"0.12rem 0.4rem",color:C.muted,fontSize:"0.62rem",cursor:"pointer"}},"삭제")
    )
  );

  return React.createElement("div",{style:{background:C.surface,border:`1px solid ${open?ac:C.border}`,borderRadius:10,overflow:"hidden",transition:"border-color 0.2s",animation:"fadeIn 0.2s ease"}},
    React.createElement("div",{style:{display:"flex",alignItems:"center",gap:6,padding:"0.48rem 0.7rem",background:C.surface2,cursor:"pointer"},onClick:()=>setOpen(!open)},
      React.createElement("div",{style:{width:18,height:18,borderRadius:4,background:ac,display:"flex",alignItems:"center",justifyContent:"center",fontSize:"0.65rem",flexShrink:0}},m?m.icon:"?"),
      React.createElement("span",{style:{fontWeight:700,fontSize:"0.77rem",color:ac,flex:1}},`${index+1}. ${m?m.label:step.type}`),
      React.createElement("div",{style:{display:"flex",gap:2},onClick:e=>e.stopPropagation()},
        React.createElement("button",{onClick:()=>onMove(index,-1),disabled:index===0,style:{background:"none",border:`1px solid ${C.border}`,borderRadius:3,padding:"0.08rem 0.28rem",color:index===0?C.border:C.muted,cursor:index===0?"default":"pointer",fontSize:"0.6rem"}},"▲"),
        React.createElement("button",{onClick:()=>onMove(index,1),disabled:index===totalSteps-1,style:{background:"none",border:`1px solid ${C.border}`,borderRadius:3,padding:"0.08rem 0.28rem",color:index===totalSteps-1?C.border:C.muted,cursor:index===totalSteps-1?"default":"pointer",fontSize:"0.6rem"}},"▼"),
        React.createElement("button",{onClick:onRemove,style:{background:"none",border:"none",color:C.muted,cursor:"pointer",fontSize:"0.82rem",padding:"0 1px",lineHeight:1}},"✕")
      ),
      React.createElement("span",{style:{color:C.muted,fontSize:"0.65rem",marginLeft:2}},open?"▲":"▼")
    ),
    open&&React.createElement("div",{style:{padding:"0.7rem"}},body)
  );
}

// ── 부가 페이지 ───────────────────────────────
function PrivacyPage(){
  return React.createElement("div",{style:{maxWidth:720,margin:"0 auto",padding:"2rem 1rem",lineHeight:1.8}},
    React.createElement("h1",{style:{fontSize:"1.6rem",fontWeight:800,marginBottom:"0.4rem"}},"개인정보처리방침"),
    React.createElement("p",{style:{color:C.muted,fontSize:"0.82rem",marginBottom:"1.75rem"}},"시행일: 2025년 1월 1일"),
    ...[
      ["1. 수집하는 개인정보","PixelForge는 별도의 개인정보를 수집하지 않습니다. 업로드된 이미지는 사용자의 브라우저 내에서만 처리되며 서버로 전송되지 않습니다."],
      ["2. 이미지 데이터 처리","업로드한 이미지는 브라우저의 Canvas API를 통해 로컬에서만 처리됩니다. 어떠한 이미지도 외부 서버에 저장되거나 전송되지 않습니다."],
      ["3. 쿠키 및 광고","Google AdSense 광고 서비스를 통해 광고 관련 쿠키가 사용될 수 있습니다. Google Analytics를 통해 익명화된 방문 통계를 수집할 수 있습니다."],
      ["4. 제3자 공유","수집된 어떠한 정보도 제3자에게 판매하거나 공유하지 않습니다."],
      ["5. 문의","개인정보 처리에 관한 문의는 문의 페이지를 이용해 주세요."],
    ].map(([t,c])=>React.createElement(React.Fragment,{key:t},
      React.createElement("h2",{style:{fontSize:"1rem",fontWeight:700,margin:"1.2rem 0 0.3rem",color:C.accent}},t),
      React.createElement("p",{style:{fontSize:"0.87rem",color:C.muted}},c)
    ))
  );
}
function TermsPage(){
  return React.createElement("div",{style:{maxWidth:720,margin:"0 auto",padding:"2rem 1rem",lineHeight:1.8}},
    React.createElement("h1",{style:{fontSize:"1.6rem",fontWeight:800,marginBottom:"0.4rem"}},"이용약관"),
    React.createElement("p",{style:{color:C.muted,fontSize:"0.82rem",marginBottom:"1.75rem"}},"최종 수정일: 2025년 1월"),
    ...[
      ["제1조 목적","본 약관은 PixelForge가 제공하는 이미지 편집 서비스 이용에 관한 조건 및 절차를 규정함을 목적으로 합니다."],
      ["제2조 서비스 내용","PixelForge는 이미지 자르기, 리사이즈, 회전, 워터마크, 얼굴 흐리기 기능을 브라우저 내에서 무료로 제공합니다."],
      ["제3조 이용자 의무","타인의 저작권·초상권을 침해하는 이미지를 편집하거나 배포해서는 안 됩니다. 불법적인 목적으로 서비스를 이용해서는 안 됩니다."],
      ["제4조 면책조항","서비스는 이미지 내용에 대해 책임을 지지 않습니다. 중요한 원본 파일은 반드시 백업하시기 바랍니다."],
      ["제5조 준거법","본 약관은 대한민국 법률에 따라 해석되고 적용됩니다."],
    ].map(([t,c])=>React.createElement(React.Fragment,{key:t},
      React.createElement("h2",{style:{fontSize:"1rem",fontWeight:700,margin:"1.2rem 0 0.3rem",color:C.accent}},t),
      React.createElement("p",{style:{fontSize:"0.87rem",color:C.muted}},c)
    ))
  );
}
function AboutPage(){
  return React.createElement("div",{style:{maxWidth:720,margin:"0 auto",padding:"2rem 1rem"}},
    React.createElement("h1",{style:{fontSize:"1.6rem",fontWeight:800,marginBottom:"0.4rem"}},"PixelForge 소개"),
    React.createElement("p",{style:{color:C.muted,marginBottom:"1.75rem",fontSize:"0.9rem",lineHeight:1.8}},"설치 없이 브라우저에서 바로 사용하는 무료 이미지 편집 도구입니다. 모든 처리는 기기 내에서만 이루어져 안전합니다."),
    React.createElement("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(200px,1fr))",gap:"0.75rem"}},
      [{icon:"✂️",t:"자르기",d:"드래그 또는 비율 고정 자르기"},{icon:"📐",t:"리사이즈",d:"배율·px 입력, Bicubic 보간"},{icon:"🔄",t:"회전/반전",d:"자유 각도 회전, 좌우·상하 반전"},{icon:"💧",t:"워터마크",d:"투명도·크기·위치 조절"},{icon:"👤",t:"얼굴 흐리기",d:"자동 감지 또는 수동 드래그"},{icon:"🔒",t:"완전 로컬",d:"이미지가 서버로 전송되지 않음"}].map(f=>
        React.createElement("div",{key:f.t,style:{background:C.surface,border:`1px solid ${C.border}`,borderRadius:10,padding:"1rem"}},
          React.createElement("div",{style:{fontSize:"1.3rem",marginBottom:"0.4rem"}},f.icon),
          React.createElement("div",{style:{fontWeight:700,fontSize:"0.88rem",marginBottom:"0.25rem"}},f.t),
          React.createElement("div",{style:{fontSize:"0.78rem",color:C.muted}},f.d)
        )
      )
    )
  );
}
function ContactPage(){
  const [name,setName]=useState(""),email=useState(""),msg=useState(""),sent=useState(false);
  const [em,setEm]=email,setMsg=msg[1],setSent=sent[1];
  const eVal=email[0],mVal=msg[0],sentVal=sent[0];
  function send(){
    if(!name||!eVal||!mVal){alert("모든 항목을 입력해주세요.");return;}
    window.location.href=`mailto:contact@pixelforge.app?subject=${encodeURIComponent("[문의] "+name)}&body=${encodeURIComponent("이름: "+name+"\n이메일: "+eVal+"\n\n"+mVal)}`;
    setSent(true);
  }
  const inp={width:"100%",background:C.surface2,border:`1px solid ${C.border}`,borderRadius:8,padding:"0.6rem 0.85rem",color:C.text,fontSize:"0.87rem",outline:"none",fontFamily:"inherit",marginBottom:"0.65rem"};
  return React.createElement("div",{style:{maxWidth:560,margin:"0 auto",padding:"2rem 1rem"}},
    React.createElement("h1",{style:{fontSize:"1.6rem",fontWeight:800,marginBottom:"0.4rem"}},"문의하기"),
    React.createElement("p",{style:{color:C.muted,fontSize:"0.88rem",marginBottom:"1.5rem"}},"버그 신고, 기능 제안, 기타 문의사항을 남겨주세요."),
    sentVal
      ?React.createElement("div",{style:{background:C.surface,border:`1px solid ${C.green}`,borderRadius:12,padding:"2rem",textAlign:"center"}},
          React.createElement("div",{style:{fontSize:"2rem",marginBottom:"0.6rem"}},"✅"),
          React.createElement("div",{style:{fontWeight:700,marginBottom:"0.3rem"}},"이메일 앱이 열렸어요!"),
          React.createElement("div",{style:{fontSize:"0.83rem",color:C.muted}},"내용을 확인 후 전송해주세요.")
        )
      :React.createElement("div",{style:{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"1.25rem"}},
          React.createElement(Label,null,"이름"),React.createElement("input",{style:inp,value:name,onChange:e=>setName(e.target.value),placeholder:"홍길동"}),
          React.createElement(Label,null,"이메일"),React.createElement("input",{style:inp,type:"email",value:eVal,onChange:e=>setEm(e.target.value),placeholder:"example@email.com"}),
          React.createElement(Label,null,"문의 내용"),React.createElement("textarea",{style:{...inp,height:120,resize:"vertical"},value:mVal,onChange:e=>setMsg(e.target.value),placeholder:"문의 내용을 입력해주세요..."}),
          React.createElement("button",{onClick:send,style:{width:"100%",padding:"0.78rem",background:`linear-gradient(135deg,${C.accent},${C.accent2})`,border:"none",borderRadius:10,color:"#fff",fontSize:"0.88rem",fontWeight:700,cursor:"pointer",fontFamily:"inherit"}},"문의 보내기 →")
        )
  );
}

// ── 메인 앱 ───────────────────────────────────
function App(){
  const [page,setPage]=useState("editor");
  const [menuOpen,setMenuOpen]=useState(false);
  const [img,setImg]=useState(null);
  const [dragging,setDragging]=useState(false);
  const [steps,setSteps]=useState([]);
  const [format,setFormat]=useState("png");
  const [quality,setQuality]=useState(92);
  const [previewUrl,setPreviewUrl]=useState(null);
  const [previewInfo,setPreviewInfo]=useState(null);
  const [previewProcessing,setPreviewProcessing]=useState(false);
  const [downloading,setDownloading]=useState(false);
  const fileRef=useRef();
  const timerRef=useRef(null);

  const updatePreview=useCallback(async(ci,cs)=>{
    if(!ci||cs.length===0){setPreviewUrl(null);setPreviewInfo(null);return;}
    setPreviewProcessing(true);
    try{const rc=await runPipeline(ci.el,cs);setPreviewUrl(rc.toDataURL("image/jpeg",0.75));setPreviewInfo({w:rc.width,h:rc.height});}
    catch(e){console.error(e);}
    setPreviewProcessing(false);
  },[]);

  useEffect(()=>{
    if(timerRef.current)clearTimeout(timerRef.current);
    timerRef.current=setTimeout(()=>updatePreview(img,steps),350);
    return()=>clearTimeout(timerRef.current);
  },[img,steps,updatePreview]);

  function loadFile(file){
    if(!file||!file.type.startsWith("image/"))return;
    setSteps([]);setPreviewUrl(null);
    const reader=new FileReader();
    reader.onload=e=>{
      const url=e.target.result;const image=new Image();
      image.onload=()=>setImg({url,w:image.width,h:image.height,name:file.name,size:file.size,el:image});
      image.src=url;
    };
    reader.readAsDataURL(file);
  }

  function addStep(type){setSteps(prev=>[...prev,defaultStep(type,img)]);}
  function updateStep(id,u){setSteps(prev=>prev.map(s=>s.id===id?u:s));}
  function removeStep(id){setSteps(prev=>prev.filter(s=>s.id!==id));}
  function moveStep(idx,dir){setSteps(prev=>{const a=[...prev],t=idx+dir;if(t<0||t>=a.length)return a;[a[idx],a[t]]=[a[t],a[idx]];return a;});}

  async function download(){
    if(!img||steps.length===0)return;
    setDownloading(true);
    try{
      const rc=await runPipeline(img.el,steps);
      const mimes={png:"image/png",jpg:"image/jpeg",webp:"image/webp"};
      const url=rc.toDataURL(mimes[format]||"image/png",format==="png"?undefined:quality/100);
      const a=document.createElement("a");a.download=img.name.replace(/\.[^.]+$/,`_pixelforge.${format}`);a.href=url;a.click();
    }catch{alert("다운로드 중 오류가 발생했어요.");}
    setDownloading(false);
  }

  const CHECKER=`repeating-conic-gradient(${C.surface3} 0% 25%,${C.surface2} 0% 50%) 0 0/16px 16px`;
  const NAV=[{id:"editor",l:"편집기"},{id:"about",l:"소개"},{id:"contact",l:"문의"},{id:"privacy",l:"개인정보"},{id:"terms",l:"이용약관"}];

  const header=React.createElement("header",{style:{height:48,display:"flex",alignItems:"center",padding:"0 1rem",borderBottom:`1px solid ${C.border}`,background:"rgba(10,10,12,0.95)",backdropFilter:"blur(12px)",position:"sticky",top:0,zIndex:200,flexShrink:0}},
    React.createElement("div",{onClick:()=>{setPage("editor");setMenuOpen(false);},style:{display:"flex",alignItems:"center",gap:7,cursor:"pointer",marginRight:"auto"}},
      React.createElement("div",{style:{width:24,height:24,borderRadius:6,background:`linear-gradient(135deg,${C.accent},${C.accent2})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:"0.8rem"}},"⬡"),
      React.createElement("span",{style:{fontWeight:800,fontSize:"0.92rem",letterSpacing:"-0.03em"}},"PixelForge")
    ),
    // 데스크탑 nav
    React.createElement("nav",{style:{display:"flex",gap:2}},
      NAV.map(l=>React.createElement("button",{key:l.id,onClick:()=>setPage(l.id),style:{padding:"0.26rem 0.62rem",borderRadius:6,border:"none",background:page===l.id?C.surface2:"transparent",color:page===l.id?C.text:C.muted,fontSize:"0.73rem",fontWeight:page===l.id?600:400,cursor:"pointer",fontFamily:"inherit"}},(window.innerWidth<600&&l.id!=="editor"&&l.id!=="about")?null:l.l))
    ),
    // 모바일 햄버거
    React.createElement("button",{onClick:()=>setMenuOpen(!menuOpen),style:{display:"none",background:"none",border:"none",color:C.muted,fontSize:"1.2rem",cursor:"pointer",padding:"0 0.25rem",marginLeft:"0.5rem",lineHeight:1},"data-mobile-menu":true},"☰")
  );

  // 모바일 드롭다운 메뉴
  const mobileMenu=menuOpen&&React.createElement("div",{style:{position:"fixed",top:48,left:0,right:0,background:C.surface,borderBottom:`1px solid ${C.border}`,zIndex:199,padding:"0.5rem"}},
    NAV.map(l=>React.createElement("button",{key:l.id,onClick:()=>{setPage(l.id);setMenuOpen(false);},style:{display:"block",width:"100%",padding:"0.65rem 1rem",borderRadius:8,border:"none",background:page===l.id?C.surface2:"transparent",color:page===l.id?C.text:C.muted,fontSize:"0.88rem",fontWeight:page===l.id?600:400,cursor:"pointer",fontFamily:"inherit",textAlign:"left"}},l.l))
  );

  // 편집기
  const editorPage=React.createElement("div",{style:{flex:1,display:"flex",flexDirection:"column",overflow:"hidden"}},
    !img
      // 업로드
      ?React.createElement("div",{style:{flex:1,display:"flex",alignItems:"center",justifyContent:"center",padding:"1rem"},
          onDragOver:e=>{e.preventDefault();setDragging(true);},onDragLeave:()=>setDragging(false),
          onDrop:e=>{e.preventDefault();setDragging(false);loadFile(e.dataTransfer.files[0]);}},
          React.createElement("div",{onClick:()=>fileRef.current.click(),style:{border:`2px dashed ${dragging?C.accent:C.border}`,borderRadius:16,padding:"3rem 2rem",textAlign:"center",cursor:"pointer",background:dragging?`${C.accent}08`:C.surface,maxWidth:480,width:"100%",transition:"all 0.2s"}},
            React.createElement("div",{style:{fontSize:"3rem",marginBottom:"0.75rem"}},"🖼️"),
            React.createElement("div",{style:{fontWeight:700,fontSize:"1rem",marginBottom:"0.4rem"}},"이미지 업로드"),
            React.createElement("div",{style:{color:C.muted,fontSize:"0.8rem",lineHeight:1.6}},"클릭하거나 파일을 여기에 드래그하세요",React.createElement("br"),"PNG · JPG · WebP · GIF · 브라우저 내 처리 🔒"),
            React.createElement("input",{ref:fileRef,type:"file",accept:"image/*",style:{display:"none"},onChange:e=>loadFile(e.target.files[0])})
          )
        )
      // 편집 레이아웃
      :React.createElement("div",{className:"editor-layout",style:{flex:1,display:"flex",overflow:"hidden"}},

          // ── 미리보기 (왼쪽/위) ──
          React.createElement("div",{className:"preview-area",style:{flex:1,display:"flex",flexDirection:"column",overflow:"hidden",minHeight:0}},
            // 툴바
            React.createElement("div",{style:{display:"flex",alignItems:"center",gap:8,padding:"0.5rem 0.75rem",borderBottom:`1px solid ${C.border}`,flexShrink:0,flexWrap:"wrap",gap:"0.4rem"}},
              React.createElement("span",{style:{fontSize:"0.7rem",color:C.muted,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:160}},img.name),
              React.createElement("span",{style:{fontSize:"0.68rem",color:C.muted}}),
              previewInfo&&steps.length>0&&React.createElement("span",{style:{fontSize:"0.68rem",color:C.muted}},`${previewInfo.w}×${previewInfo.h}px`),
              previewProcessing&&React.createElement("div",{style:{width:10,height:10,border:`2px solid ${C.border}`,borderTopColor:C.yellow,borderRadius:"50%",animation:"spin 0.6s linear infinite",flexShrink:0}}),
              React.createElement("div",{style:{marginLeft:"auto",display:"flex",gap:5,alignItems:"center"}},
                // 형식 선택
                React.createElement("div",{style:{display:"flex",gap:3}},
                  ["png","jpg","webp"].map(f=>React.createElement("button",{key:f,onClick:()=>setFormat(f),style:{padding:"0.2rem 0.45rem",borderRadius:5,border:`1px solid ${format===f?C.accent:C.border}`,background:format===f?C.accent:C.surface3,color:format===f?"#fff":C.muted,fontSize:"0.65rem",fontWeight:600,cursor:"pointer",fontFamily:"inherit"}},f.toUpperCase()))
                ),
                // 다운로드
                React.createElement("button",{onClick:download,disabled:steps.length===0||downloading,style:{padding:"0.28rem 0.85rem",borderRadius:7,border:"none",background:steps.length===0||downloading?C.surface3:`linear-gradient(135deg,${C.accent},${C.accent2})`,color:steps.length===0||downloading?C.muted:"#fff",fontSize:"0.73rem",fontWeight:700,cursor:steps.length===0||downloading?"not-allowed":"pointer",opacity:steps.length===0?0.5:1,fontFamily:"inherit",whiteSpace:"nowrap"}},
                  downloading?"저장 중...":`⬇ 다운로드`
                ),
                React.createElement("button",{onClick:()=>{setImg(null);setSteps([]);setPreviewUrl(null);},style:{padding:"0.28rem 0.6rem",borderRadius:7,border:`1px solid ${C.border}`,background:C.surface3,color:C.muted,fontSize:"0.7rem",cursor:"pointer",fontFamily:"inherit"}},"✕")
              )
            ),
            // 이미지 뷰
            React.createElement("div",{style:{flex:1,background:CHECKER,display:"flex",alignItems:"center",justifyContent:"center",overflow:"hidden",position:"relative"}},
              previewProcessing&&React.createElement("div",{style:{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",background:"rgba(10,10,12,0.5)",zIndex:2}},
                React.createElement("div",{style:{textAlign:"center"}},
                  React.createElement("div",{style:{width:28,height:28,border:`2px solid ${C.border}`,borderTopColor:C.accent,borderRadius:"50%",animation:"spin 0.6s linear infinite",margin:"0 auto 8px"}}),
                  React.createElement("div",{style:{fontSize:"0.72rem",color:C.muted}},"편집 적용 중...")
                )
              ),
              React.createElement("img",{
                src:previewUrl||img.url,
                style:{maxWidth:"100%",maxHeight:"100%",objectFit:"contain",display:"block",opacity:previewProcessing?0.3:1,transition:"opacity 0.2s"}
              }),
              !previewUrl&&steps.length===0&&React.createElement("div",{style:{position:"absolute",bottom:12,left:"50%",transform:"translateX(-50%)",background:"rgba(0,0,0,0.6)",borderRadius:8,padding:"0.4rem 0.9rem",fontSize:"0.72rem",color:C.muted,whiteSpace:"nowrap"}},
                "오른쪽에서 편집을 추가해보세요"
              )
            ),
            // JPG/WebP 품질
            format!=="png"&&React.createElement("div",{style:{padding:"0.4rem 0.75rem",borderTop:`1px solid ${C.border}`,display:"flex",alignItems:"center",gap:8,flexShrink:0}},
              React.createElement("span",{style:{fontSize:"0.68rem",color:C.muted,whiteSpace:"nowrap"}},`품질 ${quality}%`),
              React.createElement("input",{type:"range",min:10,max:100,value:quality,onChange:e=>setQuality(parseInt(e.target.value)),style:{flex:1,accentColor:C.accent,cursor:"pointer"}})
            )
          ),

          // ── 사이드바 (오른쪽/아래) ──
          React.createElement("div",{className:"sidebar",style:{width:260,borderLeft:`1px solid ${C.border}`,display:"flex",flexDirection:"column",overflow:"hidden",flexShrink:0}},
            // 편집 추가 버튼
            React.createElement("div",{style:{padding:"0.6rem",borderBottom:`1px solid ${C.border}`,flexShrink:0}},
              React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4}},
                Object.entries(STEP_META).map(([type,m])=>
                  React.createElement("button",{key:type,onClick:()=>addStep(type),style:{display:"flex",alignItems:"center",gap:5,padding:"0.42rem 0.6rem",borderRadius:8,border:`1px solid ${C.border}`,background:C.surface2,color:C.text,fontSize:"0.73rem",fontWeight:500,cursor:"pointer",textAlign:"left",transition:"all 0.12s",fontFamily:"inherit"},
                    onMouseEnter:e=>{e.currentTarget.style.borderColor=m.color;e.currentTarget.style.background=`${m.color}14`;},
                    onMouseLeave:e=>{e.currentTarget.style.borderColor=C.border;e.currentTarget.style.background=C.surface2;}
                  },React.createElement("span",{style:{fontSize:"0.85rem"}},m.icon),React.createElement("span",null,m.label))
                )
              )
            ),
            // 편집 스텝 목록
            React.createElement("div",{style:{flex:1,overflowY:"auto",padding:"0.6rem",display:"flex",flexDirection:"column",gap:"0.5rem"}},
              steps.length===0
                ?React.createElement("div",{style:{textAlign:"center",padding:"2rem 1rem",color:C.muted,fontSize:"0.78rem",lineHeight:1.7}},"위에서 편집 기능을 추가하면",React.createElement("br"),"실시간으로 미리볼 수 있어요 ✨")
                :steps.map((step,i)=>React.createElement(StepPanel,{key:step.id,step,img,index:i,totalSteps:steps.length,onChange:u=>updateStep(step.id,u),onRemove:()=>removeStep(step.id),onMove:moveStep}))
            )
          )
        )
  );

  const footer=React.createElement("footer",{style:{borderTop:`1px solid ${C.border}`,padding:"0.75rem 1rem",display:"flex",justifyContent:"center",gap:"1.5rem",flexWrap:"wrap",flexShrink:0}},
    NAV.filter(l=>l.id!=="editor").map(l=>
      React.createElement("button",{key:l.id,onClick:()=>setPage(l.id),style:{background:"none",border:"none",color:C.muted,fontSize:"0.72rem",cursor:"pointer",fontFamily:"inherit"}},l.l)
    ),
    React.createElement("span",{style:{color:C.border,fontSize:"0.72rem"}},"© 2025 PixelForge")
  );

  return React.createElement("div",{style:{background:C.bg,color:C.text,height:"100vh",display:"flex",flexDirection:"column",fontFamily:"'Inter',system-ui,sans-serif",fontSize:14,overflow:"hidden"}},
    React.createElement("style",null,globalStyle),
    header,
    mobileMenu,
    React.createElement("main",{style:{flex:1,overflow:page==="editor"?"hidden":"auto",display:"flex",flexDirection:"column"}},
      page==="editor"&&editorPage,
      page==="about"&&React.createElement(AboutPage),
      page==="privacy"&&React.createElement(PrivacyPage),
      page==="terms"&&React.createElement(TermsPage),
      page==="contact"&&React.createElement(ContactPage),
    ),
    footer
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App));
