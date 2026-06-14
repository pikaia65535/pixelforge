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
  crop:     {icon:"✂️", label:"자르기",     color:"#60c8fa"},
  resize:   {icon:"📐", label:"리사이즈",   color:"#7c6dfa"},
  rotate:   {icon:"🔄", label:"회전/반전",  color:"#4ade9a"},
  watermark:{icon:"💧", label:"워터마크",   color:"#f5c542"},
  blur:     {icon:"👤", label:"얼굴 흐리기",color:"#fa6d8a"},
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
  ::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-thumb{background:#272730;border-radius:2px;}
  @keyframes spin{to{transform:rotate(360deg);}}
  input[type=range]{width:100%;cursor:pointer;}
  body{background:#0a0a0c;}
`;

function Btn({children,onClick,active,disabled,color,small}){
  const ac=color||C.accent;
  return React.createElement("button",{onClick,disabled,style:{padding:small?"0.25rem 0.55rem":"0.37rem 0.75rem",borderRadius:7,border:`1px solid ${active?ac:C.border}`,background:active?ac:C.surface3,color:active?"#fff":C.muted,fontSize:small?"0.71rem":"0.78rem",fontWeight:600,cursor:disabled?"not-allowed":"pointer",opacity:disabled?0.4:1,transition:"all 0.15s",fontFamily:"inherit"}},children);
}
function Label({children}){return React.createElement("div",{style:{fontSize:"0.62rem",fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase",color:C.muted,marginBottom:"0.35rem"}},children);}
function Divider(){return React.createElement("hr",{style:{border:"none",borderTop:`1px solid ${C.border}`,margin:"0.7rem 0"}});}
function RangeInput({value,onChange,min,max,step,color}){return React.createElement("input",{type:"range",value,onChange,min,max,step:step||1,style:{accentColor:color||C.accent,width:"100%",cursor:"pointer"}});}
function NInput({value,onChange,min,max}){return React.createElement("input",{type:"number",value,onChange,min,max,style:{background:C.surface3,border:`1px solid ${C.border}`,borderRadius:7,padding:"0.32rem 0.5rem",color:C.text,fontSize:"0.79rem",outline:"none",fontFamily:"inherit",width:"100%"}});}

// ── 자르기 패널 ───────────────────────────────
function CropPanel({step,img,onChange}){
  const cropOverlayRef=useRef();
  const [isDrawing,setIsDrawing]=useState(false);
  const [drawStart,setDrawStart]=useState(null);
  const ac=STEP_META.crop.color;
  function upd(patch){onChange({...step,...patch});}

  function getImgRect(){
    const cv=cropOverlayRef.current;if(!cv||!img)return null;
    const rb=cv.getBoundingClientRect(),cW=rb.width,cH=rb.height;
    const iR=img.w/img.h,cR=cW/cH;
    let dW,dH,oX,oY;
    if(iR>cR){dW=cW;dH=cW/iR;oX=0;oY=(cH-dH)/2;}
    else{dH=cH;dW=cH*iR;oX=(cW-dW)/2;oY=0;}
    return{rb,dW,dH,oX,oY};
  }

  function getPos(e){
    const ir=getImgRect();if(!ir)return null;
    const{rb,dW,dH,oX,oY}=ir;
    const rX=e.clientX-rb.left-oX,rY=e.clientY-rb.top-oY;
    return{x:Math.round(Math.min(Math.max(rX,0),dW)*(img.w/dW)),y:Math.round(Math.min(Math.max(rY,0),dH)*(img.h/dH))};
  }

  function onMD(e){const p=getPos(e);if(!p)return;setIsDrawing(true);setDrawStart(p);}
  function onMM(e){
    if(!isDrawing||!drawStart)return;
    const p=getPos(e);if(!p)return;
    let x=Math.min(drawStart.x,p.x),y=Math.min(drawStart.y,p.y);
    let w=Math.abs(p.x-drawStart.x),h=Math.abs(p.y-drawStart.y);
    // 비율 고정
    if(step.aspect==="1:1"){const s=Math.min(w,h);w=s;h=s;}
    else if(step.aspect==="16:9"){h=Math.round(w*9/16);}
    else if(step.aspect==="4:3"){h=Math.round(w*3/4);}
    else if(step.aspect==="3:2"){h=Math.round(w*2/3);}
    upd({x,y,w,h});
  }
  function onMU(){setIsDrawing(false);setDrawStart(null);}

  // 화면 좌표 변환
  function cropToScreen(){
    const ir=getImgRect();if(!ir)return null;
    const{dW,dH,oX,oY}=ir;
    return{left:oX+step.x*(dW/img.w),top:oY+step.y*(dH/img.h),width:step.w*(dW/img.w),height:step.h*(dH/img.h)};
  }

  const sc=cropToScreen();

  return React.createElement(React.Fragment,null,
    React.createElement(Label,null,"비율 고정"),
    React.createElement("div",{style:{display:"flex",gap:4,flexWrap:"wrap",marginBottom:"0.7rem"}},
      ["free","1:1","16:9","4:3","3:2"].map(a=>React.createElement(Btn,{key:a,small:true,active:step.aspect===a,color:ac,onClick:()=>upd({aspect:a})},a))
    ),
    React.createElement(Label,null,"드래그로 영역 선택"),
    React.createElement("div",{style:{position:"relative",borderRadius:8,overflow:"hidden",background:`repeating-conic-gradient(${C.surface3} 0% 25%,${C.surface2} 0% 50%) 0 0/10px 10px`,userSelect:"none",marginBottom:"0.75rem"}},
      React.createElement("img",{src:img.url,style:{width:"100%",height:"auto",maxHeight:200,objectFit:"contain",display:"block"}}),
      React.createElement("canvas",{ref:cropOverlayRef,width:img.w,height:img.h,style:{position:"absolute",top:0,left:0,width:"100%",height:"100%",cursor:"crosshair"},onMouseDown:onMD,onMouseMove:onMM,onMouseUp:onMU,onMouseLeave:onMU}),
      sc&&step.w>0&&step.h>0&&React.createElement("div",{style:{position:"absolute",left:sc.left,top:sc.top,width:sc.width,height:sc.height,border:`2px solid ${ac}`,background:`${ac}18`,pointerEvents:"none"}},
        React.createElement("div",{style:{position:"absolute",top:-18,left:0,fontSize:"0.62rem",background:ac,color:"#000",padding:"0.1rem 0.4rem",borderRadius:3,fontWeight:700,whiteSpace:"nowrap"}}),
      )
    ),
    React.createElement(Label,null,"직접 입력 (px)"),
    React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:5,marginBottom:4}},
      React.createElement("div",null,React.createElement(Label,null,"X"),React.createElement(NInput,{value:step.x,min:0,max:img?img.w:9999,onChange:e=>upd({x:parseInt(e.target.value)||0})})),
      React.createElement("div",null,React.createElement(Label,null,"Y"),React.createElement(NInput,{value:step.y,min:0,max:img?img.h:9999,onChange:e=>upd({y:parseInt(e.target.value)||0})}))
    ),
    React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:5}},
      React.createElement("div",null,React.createElement(Label,null,"너비"),React.createElement(NInput,{value:step.w,min:1,max:img?img.w:9999,onChange:e=>upd({w:parseInt(e.target.value)||1})})),
      React.createElement("div",null,React.createElement(Label,null,"높이"),React.createElement(NInput,{value:step.h,min:1,max:img?img.h:9999,onChange:e=>upd({h:parseInt(e.target.value)||1})}))
    ),
    step.w>0&&step.h>0&&React.createElement("div",{style:{marginTop:"0.6rem",background:C.surface3,borderRadius:7,padding:"0.45rem 0.7rem",fontSize:"0.72rem",color:C.muted}},
      `선택 영역: `,React.createElement("span",{style:{color:C.text,fontWeight:700}},`${Math.round(step.x)},${Math.round(step.y)} — ${Math.round(step.w)}×${Math.round(step.h)}px`)
    )
  );
}

// ── 단계 패널 ─────────────────────────────────
function StepPanel({step,img,onChange,onRemove,index,totalSteps,onMove}){
  const wmFileRef=useRef();
  const blurOverlayRef=useRef();
  const [isDrawing,setIsDrawing]=useState(false);
  const [drawStart,setDrawStart]=useState(null);
  const [drawRect,setDrawRect]=useState(null);
  const [blurMode,setBlurMode]=useState("manual");
  const [faceDetecting,setFaceDetecting]=useState(false);
  const [faceApiLoaded,setFaceApiLoaded]=useState(false);
  const m=STEP_META[step.type];
  const ac=m?m.color:C.accent;
  function upd(patch){onChange({...step,...patch});}

  function getImgRect(){
    const cv=blurOverlayRef.current;if(!cv||!img)return null;
    const rb=cv.getBoundingClientRect(),cW=rb.width,cH=rb.height;
    const iR=img.w/img.h,cR=cW/cH;
    let dW,dH,oX,oY;
    if(iR>cR){dW=cW;dH=cW/iR;oX=0;oY=(cH-dH)/2;}
    else{dH=cH;dW=cH*iR;oX=(cW-dW)/2;oY=0;}
    return{rb,dW,dH,oX,oY};
  }
  function getBlurPos(e){
    const ir=getImgRect();if(!ir)return null;
    const{rb,dW,dH,oX,oY}=ir;
    const rX=e.clientX-rb.left-oX,rY=e.clientY-rb.top-oY;
    return{x:Math.round(Math.min(Math.max(rX,0),dW)*(img.w/dW)),y:Math.round(Math.min(Math.max(rY,0),dH)*(img.h/dH))};
  }
  function rectToScreen(rect){
    const ir=getImgRect();if(!ir)return null;
    const{dW,dH,oX,oY}=ir;
    return{left:oX+rect.x*(dW/img.w),top:oY+rect.y*(dH/img.h),width:rect.w*(dW/img.w),height:rect.h*(dH/img.h)};
  }
  function onMD(e){const p=getBlurPos(e);if(!p)return;setIsDrawing(true);setDrawStart(p);setDrawRect(null);}
  function onMM(e){if(!isDrawing||!drawStart)return;const p=getBlurPos(e);if(!p)return;setDrawRect({x:Math.min(drawStart.x,p.x),y:Math.min(drawStart.y,p.y),w:Math.abs(p.x-drawStart.x),h:Math.abs(p.y-drawStart.y)});}
  function onMU(){
    if(!isDrawing)return;setIsDrawing(false);
    if(drawRect&&drawRect.w>5&&drawRect.h>5)upd({rects:[...(step.rects||[]),drawRect]});
    setDrawRect(null);setDrawStart(null);
  }
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
    }catch{alert("감지 중 오류가 발생했어요.");}
    setFaceDetecting(false);
  }

  const header=React.createElement("div",{style:{display:"flex",alignItems:"center",gap:7,padding:"0.52rem 0.78rem",borderBottom:`1px solid ${C.border}`,background:C.surface2}},
    React.createElement("div",{style:{width:20,height:20,borderRadius:5,background:ac,display:"flex",alignItems:"center",justifyContent:"center",fontSize:"0.7rem",flexShrink:0}},m?m.icon:"?"),
    React.createElement("span",{style:{fontWeight:700,fontSize:"0.79rem",color:ac,flex:1}},`${index+1}. ${m?m.label:step.type}`),
    React.createElement("div",{style:{display:"flex",gap:3}},
      React.createElement("button",{onClick:()=>onMove(index,-1),disabled:index===0,style:{background:"none",border:`1px solid ${C.border}`,borderRadius:4,padding:"0.1rem 0.32rem",color:index===0?C.border:C.muted,cursor:index===0?"default":"pointer",fontSize:"0.63rem"}},"▲"),
      React.createElement("button",{onClick:()=>onMove(index,1),disabled:index===totalSteps-1,style:{background:"none",border:`1px solid ${C.border}`,borderRadius:4,padding:"0.1rem 0.32rem",color:index===totalSteps-1?C.border:C.muted,cursor:index===totalSteps-1?"default":"pointer",fontSize:"0.63rem"}},"▼"),
      React.createElement("button",{onClick:onRemove,style:{background:"none",border:"none",color:C.muted,cursor:"pointer",fontSize:"0.88rem",padding:"0 2px"}},"✕")
    )
  );

  let body=null;
  if(step.type==="crop"){
    body=React.createElement(CropPanel,{step,img,onChange});
  } else if(step.type==="resize"){
    body=React.createElement(React.Fragment,null,
      React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4,marginBottom:"0.7rem"}},
        React.createElement(Btn,{active:step.scaleMode==="ratio",color:ac,onClick:()=>upd({scaleMode:"ratio"})},"배율"),
        React.createElement(Btn,{active:step.scaleMode==="px",color:ac,onClick:()=>upd({scaleMode:"px"})},"직접 (px)")
      ),
      step.scaleMode==="ratio"
        ?React.createElement(React.Fragment,null,
            React.createElement(Label,null,`배율 ×${Number(step.ratio).toFixed(1)}`),
            React.createElement(RangeInput,{value:step.ratio,min:"0.1",max:"8",step:"0.1",color:ac,onChange:e=>upd({ratio:parseFloat(e.target.value)})}),
            React.createElement("div",{style:{display:"flex",gap:3,marginTop:"0.42rem",flexWrap:"wrap"}},
              [0.25,0.5,1,2,3,4].map(r=>React.createElement(Btn,{key:r,small:true,active:step.ratio===r,color:ac,onClick:()=>upd({ratio:r})},`×${r}`))
            )
          )
        :React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr auto 1fr",gap:4,alignItems:"end"}},
            React.createElement("div",null,React.createElement(Label,null,"너비"),React.createElement(NInput,{value:step.outW||"",min:1,max:8000,onChange:e=>{const w=parseInt(e.target.value)||1;upd({outW:w,outH:step.lockAR&&img?Math.round(w*img.h/img.w):step.outH});}})),
            React.createElement("button",{onClick:()=>upd({lockAR:!step.lockAR}),style:{background:step.lockAR?ac:C.surface3,border:`1px solid ${step.lockAR?ac:C.border}`,borderRadius:6,padding:"0.32rem",cursor:"pointer",fontSize:"0.8rem"}},step.lockAR?"🔒":"🔓"),
            React.createElement("div",null,React.createElement(Label,null,"높이"),React.createElement(NInput,{value:step.outH||"",min:1,max:8000,onChange:e=>{const h=parseInt(e.target.value)||1;upd({outH:h,outW:step.lockAR&&img?Math.round(h*img.w/img.h):step.outW});}}))
          ),
      React.createElement(Divider,null),
      React.createElement(Label,null,"보간법"),
      ...[{id:"nearest",label:"Nearest",desc:"픽셀아트"},{id:"bilinear",label:"Bilinear",desc:"4픽셀 혼합"},{id:"bicubic",label:"Bicubic ✦",desc:"최고품질"}].map(mm=>
        React.createElement("div",{key:mm.id,onClick:()=>upd({method:mm.id}),style:{padding:"0.4rem 0.62rem",borderRadius:7,cursor:"pointer",border:`1px solid ${step.method===mm.id?ac:C.border}`,background:step.method===mm.id?`${ac}18`:C.surface3,marginBottom:3,display:"flex",justifyContent:"space-between",alignItems:"center"}},
          React.createElement("span",{style:{fontWeight:600,fontSize:"0.75rem",color:step.method===mm.id?ac:C.text}},mm.label),
          React.createElement("span",{style:{fontSize:"0.63rem",color:C.muted}},mm.desc)
        )
      )
    );
  } else if(step.type==="rotate"){
    body=React.createElement(React.Fragment,null,
      React.createElement(Label,null,`회전 ${step.deg}°`),
      React.createElement(RangeInput,{value:step.deg,min:0,max:359,color:ac,onChange:e=>upd({deg:parseInt(e.target.value)})}),
      React.createElement("div",{style:{display:"flex",gap:3,marginTop:"0.42rem"}},
        [0,90,180,270].map(d=>React.createElement(Btn,{key:d,small:true,active:step.deg===d,color:ac,onClick:()=>upd({deg:d})},`${d}°`))
      ),
      React.createElement(Divider,null),
      React.createElement(Label,null,"반전"),
      React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4}},
        React.createElement(Btn,{active:step.flipH,color:ac,onClick:()=>upd({flipH:!step.flipH})},"↔ 좌우"),
        React.createElement(Btn,{active:step.flipV,color:ac,onClick:()=>upd({flipV:!step.flipV})},"↕ 상하")
      )
    );
  } else if(step.type==="watermark"){
    body=React.createElement(React.Fragment,null,
      React.createElement("div",{onClick:()=>wmFileRef.current.click(),style:{border:`2px dashed ${step.wmImg?ac:C.border}`,borderRadius:8,padding:"0.6rem",textAlign:"center",cursor:"pointer",marginBottom:"0.7rem",background:C.surface3}},
        step.wmImg
          ?React.createElement("div",{style:{display:"flex",alignItems:"center",gap:7}},
              React.createElement("img",{src:step.wmImg.url,style:{width:30,height:30,objectFit:"contain",borderRadius:4}}),
              React.createElement("div",null,React.createElement("div",{style:{fontSize:"0.72rem",fontWeight:600,color:C.green}},"✓ 로드됨"),React.createElement("div",{style:{fontSize:"0.62rem",color:C.muted}},`${step.wmImg.w}×${step.wmImg.h}`))
            )
          :React.createElement("div",{style:{fontSize:"0.74rem",color:C.muted}},"클릭해서 이미지 선택")
      ),
      React.createElement("input",{ref:wmFileRef,type:"file",accept:"image/*",style:{display:"none"},onChange:e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=ev=>{const url=ev.target.result;const im=new Image();im.onload=()=>upd({wmImg:{url,w:im.width,h:im.height,el:im}});im.src=url;};r.readAsDataURL(f);}}),
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
  } else if(step.type==="blur"){
    const allRects=[...(step.rects||[]),...(drawRect?[drawRect]:[])];
    body=React.createElement(React.Fragment,null,
      React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4,marginBottom:"0.62rem"}},
        React.createElement(Btn,{active:blurMode==="auto",color:ac,onClick:()=>setBlurMode("auto")},"🤖 자동"),
        React.createElement(Btn,{active:blurMode==="manual",color:ac,onClick:()=>setBlurMode("manual")},"✏️ 수동")
      ),
      blurMode==="auto"&&React.createElement("button",{onClick:detectFaces,disabled:faceDetecting,style:{width:"100%",padding:"0.46rem",background:faceDetecting?C.surface3:ac,border:"none",borderRadius:8,color:faceDetecting?C.muted:"#fff",fontWeight:700,fontSize:"0.76rem",cursor:"pointer",marginBottom:"0.62rem",fontFamily:"inherit"}},faceDetecting?"모델 로딩 중...":"🔍 얼굴 자동 감지"),
      blurMode==="manual"&&img&&React.createElement("div",{style:{position:"relative",marginBottom:"0.62rem",borderRadius:8,overflow:"hidden",background:`repeating-conic-gradient(${C.surface3} 0% 25%,${C.surface2} 0% 50%) 0 0/10px 10px`,userSelect:"none"}},
        React.createElement("img",{src:img.url,style:{width:"100%",height:"auto",maxHeight:130,objectFit:"contain",display:"block"}}),
        React.createElement("canvas",{ref:blurOverlayRef,width:img.w,height:img.h,style:{position:"absolute",top:0,left:0,width:"100%",height:"100%",cursor:"crosshair"},onMouseDown:onMD,onMouseMove:onMM,onMouseUp:onMU,onMouseLeave:onMU}),
        allRects.map((rect,i)=>{const s=rectToScreen(rect);if(!s)return null;return React.createElement("div",{key:i,style:{position:"absolute",left:s.left,top:s.top,width:s.width,height:s.height,border:`2px solid ${ac}`,background:`${ac}22`,borderRadius:3,pointerEvents:"none"}});})
      ),
      React.createElement(Label,null,`블러 강도 ${step.blurRadius}`),
      React.createElement(RangeInput,{value:step.blurRadius,min:5,max:50,color:ac,onChange:e=>upd({blurRadius:parseInt(e.target.value)})}),
      React.createElement("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginTop:"0.48rem"}},
        React.createElement("span",{style:{fontSize:"0.69rem",color:C.muted}},`영역 ${(step.rects||[]).length}개`),
        (step.rects||[]).length>0&&React.createElement("button",{onClick:()=>upd({rects:[]}),style:{background:"none",border:`1px solid ${C.border}`,borderRadius:5,padding:"0.14rem 0.42rem",color:C.muted,fontSize:"0.64rem",cursor:"pointer"}},"삭제")
      )
    );
  }

  return React.createElement("div",{style:{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,overflow:"hidden"}},header,React.createElement("div",{style:{padding:"0.78rem"}},body));
}

// ── 비교 뷰어 ─────────────────────────────────
function CompareViewer({originalUrl,resultUrl,originalInfo,resultInfo,processing}){
  const CHECKER=`repeating-conic-gradient(${C.surface3} 0% 25%,${C.surface2} 0% 50%) 0 0/16px 16px`;
  function Panel({label,labelColor,url,info,isProcessing}){
    return React.createElement("div",{style:{background:C.surface,border:`1px solid ${labelColor&&url?labelColor:C.border}`,borderRadius:12,overflow:"hidden",transition:"border-color 0.3s"}},
      React.createElement("div",{style:{padding:"0.48rem 0.72rem",borderBottom:`1px solid ${C.border}`,display:"flex",justifyContent:"space-between",alignItems:"center"}},
        React.createElement("span",{style:{fontSize:"0.68rem",fontWeight:700,color:labelColor||C.muted,textTransform:"uppercase",letterSpacing:"0.08em"}},label),
        info&&React.createElement("span",{style:{fontSize:"0.63rem",color:C.muted}},`${info.w}×${info.h}${info.size?" · "+fmtBytes(info.size):""}`)
      ),
      React.createElement("div",{style:{background:CHECKER,minHeight:200,display:"flex",alignItems:"center",justifyContent:"center",position:"relative"}},
        isProcessing&&React.createElement("div",{style:{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",background:"rgba(10,10,12,0.55)",zIndex:2}},
          React.createElement("div",{style:{textAlign:"center"}},
            React.createElement("div",{style:{width:26,height:26,border:`2px solid ${C.border}`,borderTopColor:C.accent,borderRadius:"50%",animation:"spin 0.6s linear infinite",margin:"0 auto 7px"}}),
            React.createElement("div",{style:{fontSize:"0.7rem",color:C.muted}},"적용 중...")
          )
        ),
        url?React.createElement("img",{src:url,style:{width:"100%",height:"auto",maxHeight:300,objectFit:"contain",display:"block",opacity:isProcessing?0.25:1,transition:"opacity 0.2s"}}):
          React.createElement("span",{style:{color:C.muted,fontSize:"0.78rem",textAlign:"center",padding:"1rem"}},isProcessing?"처리 중...":"편집을 추가하면\n여기서 미리볼 수 있어요")
      )
    );
  }
  return React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0.6rem"}},
    React.createElement(Panel,{label:"원본",url:originalUrl,info:originalInfo}),
    React.createElement(Panel,{label:processing?"처리 중...":resultUrl?"결과 미리보기":"편집 없음",labelColor:processing?C.yellow:resultUrl?C.green:null,url:resultUrl,info:resultInfo,isProcessing:processing})
  );
}

// ── 부가 페이지들 ──────────────────────────────
function PrivacyPage(){
  const today=new Date().toLocaleDateString("ko-KR");
  return React.createElement("div",{style:{maxWidth:760,margin:"0 auto",padding:"3rem 1.5rem",lineHeight:1.8,color:C.text}},
    React.createElement("h1",{style:{fontSize:"1.8rem",fontWeight:800,marginBottom:"0.5rem"}},"개인정보처리방침"),
    React.createElement("p",{style:{color:C.muted,fontSize:"0.85rem",marginBottom:"2rem"}},`시행일: ${today}`),
    ...[
      ["1. 수집하는 개인정보","PixelForge는 회원가입, 로그인 등의 서비스를 제공하지 않으며 별도의 개인정보를 수집하지 않습니다. 업로드된 이미지는 사용자의 브라우저 내에서만 처리되며 서버로 전송되지 않습니다."],
      ["2. 이미지 데이터 처리","사용자가 업로드한 이미지는 브라우저의 Canvas API를 통해 로컬에서만 처리됩니다. 어떠한 이미지도 외부 서버에 저장되거나 전송되지 않습니다."],
      ["3. 쿠키 및 추적","PixelForge는 서비스 개선을 위해 Google Analytics 등의 분석 도구를 사용할 수 있으며, 이 경우 익명화된 방문 통계만 수집됩니다. 광고 서비스(Google AdSense)를 통해 광고 관련 쿠키가 사용될 수 있습니다."],
      ["4. 제3자 공유","수집된 어떠한 정보도 제3자에게 판매하거나 공유하지 않습니다."],
      ["5. 보안","모든 이미지 처리는 사용자의 기기 내에서 이루어지므로 데이터 유출 위험이 없습니다."],
      ["6. 문의","개인정보 처리에 관한 문의는 하단 문의 페이지를 이용해 주세요."],
    ].map(([title,content])=>React.createElement(React.Fragment,{key:title},
      React.createElement("h2",{style:{fontSize:"1.05rem",fontWeight:700,margin:"1.5rem 0 0.4rem",color:C.accent}},title),
      React.createElement("p",{style:{fontSize:"0.9rem",color:C.muted}},content)
    ))
  );
}

function TermsPage(){
  return React.createElement("div",{style:{maxWidth:760,margin:"0 auto",padding:"3rem 1.5rem",lineHeight:1.8,color:C.text}},
    React.createElement("h1",{style:{fontSize:"1.8rem",fontWeight:800,marginBottom:"0.5rem"}},"이용약관"),
    React.createElement("p",{style:{color:C.muted,fontSize:"0.85rem",marginBottom:"2rem"}},"최종 수정일: 2025년 1월"),
    ...[
      ["제1조 (목적)","본 약관은 PixelForge(이하 '서비스')가 제공하는 이미지 편집 서비스 이용에 관한 조건 및 절차, 이용자와 서비스 간의 권리, 의무 및 책임사항을 규정함을 목적으로 합니다."],
      ["제2조 (서비스 내용)","PixelForge는 이미지 리사이즈, 자르기, 회전, 워터마크 추가, 얼굴 흐리기 등의 이미지 편집 기능을 브라우저 내에서 무료로 제공합니다."],
      ["제3조 (이용자의 의무)","이용자는 서비스를 통해 타인의 저작권, 초상권 등 지적재산권을 침해하는 이미지를 편집하거나 배포해서는 안 됩니다. 불법적인 목적으로 서비스를 이용해서는 안 됩니다."],
      ["제4조 (면책조항)","서비스는 이용자가 업로드한 이미지의 내용에 대해 책임을 지지 않습니다. 서비스 이용 중 발생한 데이터 손실에 대해 책임을 지지 않으므로 중요한 원본 파일은 반드시 백업하시기 바랍니다."],
      ["제5조 (서비스 변경 및 중단)","서비스는 사전 고지 없이 서비스 내용을 변경하거나 중단할 수 있습니다."],
      ["제6조 (준거법)","본 약관은 대한민국 법률에 따라 해석되고 적용됩니다."],
    ].map(([title,content])=>React.createElement(React.Fragment,{key:title},
      React.createElement("h2",{style:{fontSize:"1.05rem",fontWeight:700,margin:"1.5rem 0 0.4rem",color:C.accent}},title),
      React.createElement("p",{style:{fontSize:"0.9rem",color:C.muted}},content)
    ))
  );
}

function AboutPage(){
  return React.createElement("div",{style:{maxWidth:760,margin:"0 auto",padding:"3rem 1.5rem",color:C.text}},
    React.createElement("h1",{style:{fontSize:"1.8rem",fontWeight:800,letterSpacing:"-0.04em",marginBottom:"0.5rem"}},"PixelForge 소개"),
    React.createElement("p",{style:{color:C.muted,marginBottom:"2.5rem",fontSize:"0.95rem",lineHeight:1.8}},"PixelForge는 별도 설치 없이 브라우저에서 바로 사용할 수 있는 무료 이미지 편집 도구입니다."),
    React.createElement("div",{style:{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:"1rem",marginBottom:"2.5rem"}},
      ...[
        {icon:"✂️",title:"자르기",desc:"원하는 영역을 드래그로 선택하거나 비율을 고정해 정확하게 자를 수 있어요."},
        {icon:"📐",title:"리사이즈",desc:"배율 또는 직접 px 입력으로 크기를 조절합니다. Bicubic 보간으로 최고 품질을 제공해요."},
        {icon:"🔄",title:"회전/반전",desc:"자유로운 각도 회전과 좌우·상하 반전을 지원합니다."},
        {icon:"💧",title:"워터마크",desc:"로고나 이미지를 투명도와 위치를 조절하여 삽입할 수 있어요."},
        {icon:"👤",title:"얼굴 흐리기",desc:"자동 얼굴 감지 또는 수동 드래그로 원하는 영역을 흐릴 수 있습니다."},
        {icon:"🔒",title:"완전 로컬 처리",desc:"모든 편집은 브라우저 내에서만 처리됩니다. 이미지가 서버로 전송되지 않아요."},
      ].map(f=>React.createElement("div",{key:f.title,style:{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"1.25rem"}},
        React.createElement("div",{style:{fontSize:"1.5rem",marginBottom:"0.5rem"}},f.icon),
        React.createElement("div",{style:{fontWeight:700,marginBottom:"0.35rem"}},f.title),
        React.createElement("div",{style:{fontSize:"0.82rem",color:C.muted,lineHeight:1.6}},f.desc)
      ))
    ),
    React.createElement("div",{style:{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"1.5rem"}},
      React.createElement("h2",{style:{fontWeight:700,marginBottom:"0.5rem"}},"왜 PixelForge인가요?"),
      React.createElement("p",{style:{fontSize:"0.88rem",color:C.muted,lineHeight:1.8}},"기존 이미지 편집 도구들은 설치가 필요하거나 파일을 서버에 업로드해야 했어요. PixelForge는 모든 처리를 브라우저 내에서 완료하기 때문에 개인 이미지를 안전하게 편집할 수 있습니다. 여러 편집을 순서대로 쌓아서 한 번에 적용하는 파이프라인 방식으로, 복잡한 편집도 간편하게 처리할 수 있어요.")
    )
  );
}

function ContactPage(){
  const [name,setName]=useState("");
  const [email,setEmail]=useState("");
  const [msg,setMsg]=useState("");
  const [sent,setSent]=useState(false);

  function handleSubmit(){
    if(!name||!email||!msg){alert("모든 항목을 입력해주세요.");return;}
    // mailto 방식 (서버 없이)
    const subject=encodeURIComponent(`[PixelForge 문의] ${name}`);
    const body=encodeURIComponent(`이름: ${name}\n이메일: ${email}\n\n내용:\n${msg}`);
    window.location.href=`mailto:contact@pixelforge.app?subject=${subject}&body=${body}`;
    setSent(true);
  }

  const inputStyle={width:"100%",background:C.surface3,border:`1px solid ${C.border}`,borderRadius:8,padding:"0.65rem 0.9rem",color:C.text,fontSize:"0.88rem",outline:"none",fontFamily:"inherit",marginBottom:"0.75rem",boxSizing:"border-box"};

  return React.createElement("div",{style:{maxWidth:600,margin:"0 auto",padding:"3rem 1.5rem",color:C.text}},
    React.createElement("h1",{style:{fontSize:"1.8rem",fontWeight:800,letterSpacing:"-0.04em",marginBottom:"0.5rem"}},"문의하기"),
    React.createElement("p",{style:{color:C.muted,fontSize:"0.9rem",marginBottom:"2rem",lineHeight:1.7}},"버그 신고, 기능 제안, 기타 문의사항을 남겨주세요. 빠르게 답변 드리겠습니다."),
    sent
      ?React.createElement("div",{style:{background:C.surface,border:`1px solid ${C.green}`,borderRadius:12,padding:"2rem",textAlign:"center"}},
          React.createElement("div",{style:{fontSize:"2rem",marginBottom:"0.75rem"}},"✅"),
          React.createElement("div",{style:{fontWeight:700,marginBottom:"0.4rem"}},"이메일 앱이 열렸어요!"),
          React.createElement("div",{style:{fontSize:"0.85rem",color:C.muted}},"내용을 확인하고 전송해 주세요.")
        )
      :React.createElement("div",{style:{background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,padding:"1.5rem"}},
          React.createElement("div",{style:{fontSize:"0.72rem",fontWeight:700,color:C.muted,textTransform:"uppercase",letterSpacing:"0.08em",marginBottom:"0.35rem"}},"이름"),
          React.createElement("input",{style:inputStyle,value:name,onChange:e=>setName(e.target.value),placeholder:"홍길동"}),
          React.createElement("div",{style:{fontSize:"0.72rem",fontWeight:700,color:C.muted,textTransform:"uppercase",letterSpacing:"0.08em",marginBottom:"0.35rem"}},"이메일"),
          React.createElement("input",{style:inputStyle,type:"email",value:email,onChange:e=>setEmail(e.target.value),placeholder:"example@email.com"}),
          React.createElement("div",{style:{fontSize:"0.72rem",fontWeight:700,color:C.muted,textTransform:"uppercase",letterSpacing:"0.08em",marginBottom:"0.35rem"}},"문의 내용"),
          React.createElement("textarea",{style:{...inputStyle,height:130,resize:"vertical"},value:msg,onChange:e=>setMsg(e.target.value),placeholder:"문의 내용을 자세히 작성해주세요..."}),
          React.createElement("button",{onClick:handleSubmit,style:{width:"100%",padding:"0.82rem",background:`linear-gradient(135deg,${C.accent},${C.accent2})`,border:"none",borderRadius:10,color:"#fff",fontSize:"0.9rem",fontWeight:700,cursor:"pointer",fontFamily:"inherit"}},"문의 보내기 →")
        )
  );
}

// ── 메인 앱 ───────────────────────────────────
function App(){
  const [page,setPage]=useState("editor"); // editor | privacy | terms | about | contact
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

  const updatePreview=useCallback(async(currentImg,currentSteps)=>{
    if(!currentImg||currentSteps.length===0){setPreviewUrl(null);setPreviewInfo(null);return;}
    setPreviewProcessing(true);
    try{
      const rc=await runPipeline(currentImg.el,currentSteps);
      setPreviewUrl(rc.toDataURL("image/jpeg",0.75));
      setPreviewInfo({w:rc.width,h:rc.height});
    }catch(e){console.error(e);}
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
  function updateStep(id,updated){setSteps(prev=>prev.map(s=>s.id===id?updated:s));}
  function removeStep(id){setSteps(prev=>prev.filter(s=>s.id!==id));}
  function moveStep(idx,dir){
    setSteps(prev=>{
      const arr=[...prev],t=idx+dir;
      if(t<0||t>=arr.length)return arr;
      [arr[idx],arr[t]]=[arr[t],arr[idx]];return arr;
    });
  }

  async function downloadResult(){
    if(!img||steps.length===0)return;
    setDownloading(true);
    try{
      const rc=await runPipeline(img.el,steps);
      const mimes={png:"image/png",jpg:"image/jpeg",webp:"image/webp"};
      const dataUrl=rc.toDataURL(mimes[format]||"image/png",format==="png"?undefined:quality/100);
      const a=document.createElement("a");
      a.download=img.name.replace(/\.[^.]+$/,`_pixelforge.${format}`);
      a.href=dataUrl;a.click();
    }catch(e){alert("다운로드 중 오류가 발생했어요.");}
    setDownloading(false);
  }

  const NAV_LINKS=[
    {id:"editor",label:"편집기"},
    {id:"about",label:"소개"},
    {id:"contact",label:"문의"},
    {id:"privacy",label:"개인정보처리방침"},
    {id:"terms",label:"이용약관"},
  ];

  // 헤더
  const header=React.createElement("div",{style:{borderBottom:`1px solid ${C.border}`,padding:"0 1.5rem",height:50,display:"flex",alignItems:"center",gap:12,background:"rgba(10,10,12,0.95)",backdropFilter:"blur(12px)",position:"sticky",top:0,zIndex:100,flexWrap:"wrap"}},
    React.createElement("div",{onClick:()=>setPage("editor"),style:{display:"flex",alignItems:"center",gap:8,cursor:"pointer"}},
      React.createElement("div",{style:{width:26,height:26,borderRadius:7,background:`linear-gradient(135deg,${C.accent},${C.accent2})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:"0.85rem"}},"⬡"),
      React.createElement("span",{style:{fontWeight:800,fontSize:"0.95rem",letterSpacing:"-0.03em"}}),"PixelForge"
    ),
    React.createElement("div",{style:{display:"flex",gap:2,marginLeft:"auto"}},
      NAV_LINKS.map(l=>React.createElement("button",{key:l.id,onClick:()=>setPage(l.id),style:{padding:"0.28rem 0.7rem",borderRadius:6,border:"none",background:page===l.id?C.surface2:"transparent",color:page===l.id?C.text:C.muted,fontSize:"0.75rem",fontWeight:page===l.id?600:400,cursor:"pointer",fontFamily:"inherit",transition:"all 0.15s"}},l.label))
    )
  );

  // 편집기 페이지
  const editorPage=React.createElement("div",{style:{maxWidth:1100,margin:"0 auto",padding:"1.1rem 1rem"}},
    !img
      ?React.createElement("div",{onDragOver:e=>{e.preventDefault();setDragging(true);},onDragLeave:()=>setDragging(false),onDrop:e=>{e.preventDefault();setDragging(false);loadFile(e.dataTransfer.files[0]);},onClick:()=>fileRef.current.click(),style:{border:`2px dashed ${dragging?C.accent:C.border}`,borderRadius:18,padding:"6rem 2rem",textAlign:"center",cursor:"pointer",transition:"all 0.2s",background:dragging?`${C.accent}08`:C.surface}},
          React.createElement("div",{style:{fontSize:"3.5rem",marginBottom:"1rem"}},"🖼️"),
          React.createElement("div",{style:{fontWeight:700,fontSize:"1.05rem",marginBottom:"0.5rem"}},"이미지를 드래그하거나 클릭해서 업로드"),
          React.createElement("div",{style:{color:C.muted,fontSize:"0.8rem"}},"PNG · JPG · WebP · GIF 지원 · 브라우저에서만 처리 🔒"),
          React.createElement("input",{ref:fileRef,type:"file",accept:"image/*",style:{display:"none"},onChange:e=>loadFile(e.target.files[0])})
        )
      :React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 285px",gap:"1rem",alignItems:"start"}},
          // 왼쪽
          React.createElement("div",{style:{display:"flex",flexDirection:"column",gap:"0.72rem"}},
            React.createElement("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center"}},
              React.createElement("span",{style:{fontSize:"0.75rem",color:C.muted}}),
              React.createElement("button",{onClick:()=>{setImg(null);setSteps([]);setPreviewUrl(null);},style:{background:C.surface3,border:`1px solid ${C.border}`,borderRadius:7,padding:"0.24rem 0.58rem",color:C.muted,fontSize:"0.71rem",cursor:"pointer"}},"다른 이미지 선택")
            ),
            React.createElement(CompareViewer,{originalUrl:img.url,resultUrl:previewUrl,originalInfo:{w:img.w,h:img.h,size:img.size},resultInfo:previewInfo,processing:previewProcessing}),
            steps.length>0&&React.createElement("div",{style:{fontSize:"0.63rem",fontWeight:700,color:C.muted,textTransform:"uppercase",letterSpacing:"0.1em"}},`편집 순서 — ${steps.length}개 · 위→아래 순차 적용`),
            ...steps.map((step,i)=>React.createElement(StepPanel,{key:step.id,step,img,index:i,totalSteps:steps.length,onChange:updated=>updateStep(step.id,updated),onRemove:()=>removeStep(step.id),onMove:moveStep})),
            steps.length===0&&React.createElement("div",{style:{textAlign:"center",padding:"1.5rem",color:C.muted,fontSize:"0.8rem",background:C.surface,border:`1px dashed ${C.border}`,borderRadius:12}},"오른쪽에서 편집을 추가하면 여기서 실시간으로 확인할 수 있어요 ✨")
          ),
          // 오른쪽
          React.createElement("div",{style:{display:"flex",flexDirection:"column",gap:"0.72rem",position:"sticky",top:58}},
            React.createElement("div",{style:{background:C.surface,border:`1px solid ${C.border}`,borderRadius:13,padding:"0.88rem"}},
              React.createElement(Label,null,"편집 추가"),
              React.createElement("div",{style:{display:"flex",flexDirection:"column",gap:4}},
                Object.entries(STEP_META).map(([type,m])=>
                  React.createElement("button",{key:type,onClick:()=>addStep(type),style:{display:"flex",alignItems:"center",gap:8,padding:"0.48rem 0.68rem",borderRadius:8,border:`1px solid ${C.border}`,background:C.surface3,color:C.text,fontSize:"0.79rem",fontWeight:500,cursor:"pointer",textAlign:"left",transition:"all 0.12s",fontFamily:"inherit"},
                    onMouseEnter:e=>{e.currentTarget.style.borderColor=m.color;e.currentTarget.style.background=`${m.color}12`;},
                    onMouseLeave:e=>{e.currentTarget.style.borderColor=C.border;e.currentTarget.style.background=C.surface3;}
                  },React.createElement("span",null,m.icon),React.createElement("span",{style:{flex:1}},m.label),React.createElement("span",{style:{color:C.muted,fontSize:"0.73rem"}},"+"))
                )
              )
            ),
            React.createElement("div",{style:{background:C.surface,border:`1px solid ${C.border}`,borderRadius:13,padding:"0.88rem"}},
              React.createElement(Label,null,"저장 형식"),
              React.createElement("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:4,marginBottom:format!=="png"?"0.72rem":"0"}},
                ["png","jpg","webp"].map(f=>React.createElement(Btn,{key:f,active:format===f,color:C.accent,onClick:()=>setFormat(f)},f.toUpperCase()))
              ),
              format!=="png"&&React.createElement(React.Fragment,null,
                React.createElement(Label,null,`품질 ${quality}%`),
                React.createElement(RangeInput,{value:quality,min:10,max:100,color:C.accent,onChange:e=>setQuality(parseInt(e.target.value))}),
                React.createElement("div",{style:{display:"flex",gap:3,marginTop:"0.38rem",flexWrap:"wrap"}},
                  [60,75,85,92,100].map(q=>React.createElement(Btn,{key:q,small:true,active:quality===q,color:C.accent,onClick:()=>setQuality(q)},q))
                )
              )
            ),
            steps.length>0&&React.createElement("div",{style:{background:C.surface,border:`1px solid ${previewProcessing?C.yellow:previewUrl?C.green:C.border}`,borderRadius:10,padding:"0.62rem 0.78rem",fontSize:"0.71rem",display:"flex",alignItems:"center",gap:7,transition:"border-color 0.3s"}},
              previewProcessing
                ?React.createElement(React.Fragment,null,React.createElement("div",{style:{width:11,height:11,border:`2px solid ${C.border}`,borderTopColor:C.yellow,borderRadius:"50%",animation:"spin 0.6s linear infinite",flexShrink:0}}),React.createElement("span",{style:{color:C.yellow}},"미리보기 갱신 중..."))
                :previewUrl
                ?React.createElement(React.Fragment,null,React.createElement("span",{style:{color:C.green}},"✓"),React.createElement("span",{style:{color:C.muted}},"미리보기 최신 상태"))
                :React.createElement("span",{style:{color:C.muted}},"편집을 추가하면 자동으로 미리봐요")
            ),
            React.createElement("div",{style:{background:C.surface,border:`1px solid ${C.border}`,borderRadius:10,padding:"0.68rem 0.82rem"}},
              React.createElement("div",{style:{fontSize:"0.63rem",color:C.muted,marginBottom:"0.38rem",textTransform:"uppercase",letterSpacing:"0.08em"}},"파일 정보"),
              React.createElement("div",{style:{fontSize:"0.74rem",fontWeight:600,marginBottom:"0.18rem",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}},img.name),
              React.createElement("div",{style:{fontSize:"0.68rem",color:C.muted}},`${img.w}×${img.h}px · ${fmtBytes(img.size)}`),
              previewInfo&&steps.length>0&&React.createElement("div",{style:{borderTop:`1px solid ${C.border}`,marginTop:"0.48rem",paddingTop:"0.48rem",fontSize:"0.68rem",color:C.muted}},"편집 후: ",React.createElement("span",{style:{color:C.text,fontWeight:600}},`${previewInfo.w}×${previewInfo.h}px`))
            ),
            React.createElement("button",{onClick:downloadResult,disabled:steps.length===0||downloading,style:{width:"100%",padding:"0.85rem",background:steps.length===0||downloading?C.surface3:`linear-gradient(135deg,${C.accent},${C.accent2})`,border:"none",borderRadius:12,color:steps.length===0||downloading?C.muted:"#fff",fontSize:"0.88rem",fontWeight:700,cursor:steps.length===0||downloading?"not-allowed":"pointer",fontFamily:"inherit",transition:"all 0.2s",opacity:steps.length===0?0.5:1}},downloading?"다운로드 준비 중...":`⬇ 다운로드 (.${format.toUpperCase()})`)
          )
        )
  );

  const footer=React.createElement("div",{style:{borderTop:`1px solid ${C.border}`,padding:"1.5rem",textAlign:"center",fontSize:"0.72rem",color:C.muted,marginTop:"3rem"}},
    React.createElement("div",{style:{display:"flex",justifyContent:"center",gap:"1.5rem",flexWrap:"wrap"}},
      [{id:"about",l:"소개"},{id:"privacy",l:"개인정보처리방침"},{id:"terms",l:"이용약관"},{id:"contact",l:"문의"}].map(l=>
        React.createElement("span",{key:l.id,onClick:()=>setPage(l.id),style:{cursor:"pointer",color:C.muted,transition:"color 0.15s"},onMouseEnter:e=>e.target.style.color=C.accent,onMouseLeave:e=>e.target.style.color=C.muted},l.l)
      )
    ),
    React.createElement("div",{style:{marginTop:"0.75rem"}},"© 2025 PixelForge · 모든 이미지는 브라우저 내에서만 처리됩니다 🔒")
  );

  return React.createElement("div",{style:{background:C.bg,color:C.text,minHeight:"100vh",fontFamily:"'Inter',system-ui,sans-serif",fontSize:14}},
    React.createElement("style",null,globalStyle),
    header,
    page==="editor"&&editorPage,
    page==="about"&&React.createElement(AboutPage),
    page==="privacy"&&React.createElement(PrivacyPage),
    page==="terms"&&React.createElement(TermsPage),
    page==="contact"&&React.createElement(ContactPage),
    footer
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App));
