(() => {
  const $ = (selector) => document.querySelector(selector);
  const form = $('#form'), fileInput = $('#file'), zone = $('#dropzone'), errorBox = $('#error');
  const results = $('#analysis'), canvas = $('#histogram'), ctx = canvas.getContext('2d'), tooltip = $('#tooltip');
  const state = { file: null, data: null, history: [], index: -1, requestId: 0, objectUrl: null,
    baseConfig: null, visible: {}, range: [0, 255], drag: null, zoom: 1, panX: 0, panY: 0 };
  const methodNames = {stretch:'Kéo giãn tương phản', equalize:'Cân bằng Histogram toàn cục', clahe:'CLAHE', gamma:'Hiệu chỉnh Gamma'};
  const morphNames = {none:'Không có', erode:'Co ảnh', dilate:'Giãn ảnh', open:'Mở ảnh', close:'Đóng ảnh'};
  const shapeNames = {ellipse:'elip', rect:'chữ nhật', cross:'dấu cộng'};
  const niceBytes = n => n < 1024 ? `${n} B` : n < 1048576 ? `${(n/1024).toFixed(1)} KB` : `${(n/1048576).toFixed(2)} MB`;
  const f = n => Number(n).toLocaleString('vi-VN',{maximumFractionDigits:2});
  const img = (selector, url) => { $(selector).src = url; };
  let matchedReference = null, matchedDetails = null;

  function snapshot() {
    const fd = new FormData(form), result = {};
    for (const [key,value] of fd.entries()) if (key !== 'image') result[key] = value;
    return result;
  }
  function applySnapshot(config) { for (const [key,value] of Object.entries(config)) { const el = form.elements.namedItem(key); if (el) el.value = value; } updateMethodControls(); }
  function updateHistoryButtons() { $('#undo').disabled = !state.file || state.index <= 0; $('#redo').disabled = !state.file || state.index < 0 || state.index >= state.history.length-1; $('#reset').disabled = !state.file; const history=state.history.map((item,i)=>`${i===state.index?'●':'○'} ${item===null?'Ảnh gốc':methodNames[item.method]}`).join(' → ');$('#history-label').textContent=history||'Ảnh gốc'; }
  function resetResults() { state.data = null; results.hidden = true; $('#warning').hidden=true; }
  function setFile(file) {
    if (!file) return;
    state.requestId++;document.body.classList.remove('loading'); state.file = file; state.history=[null]; state.index=0; state.data=null;matchedReference=null;matchedDetails=null;$('#match-results').hidden=true;resetResults();
    $('#filename').textContent=file.name; $('#filesize').textContent=niceBytes(file.size); $('#filesize-meta').textContent=niceBytes(file.size);
    $('#file-row').hidden=false; zone.style.display='none'; $('#submit').disabled=false; errorBox.textContent='';
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl=URL.createObjectURL(file); $('#viewer-original').src=state.objectUrl;
    $('#viewer-result').src=state.objectUrl; $('#viewer-stack').hidden=false; $('.empty').hidden=true;
    $('#image-status').textContent='Sẵn sàng phân tích'; $('#dim').textContent='Đang chờ'; $('#channels').textContent='Đang chờ'; $('#format').textContent=(file.name.split('.').pop()||'').toUpperCase();$('#run-match').disabled=!$('#reference-file').files[0];
    state.zoom=1;state.panX=0;state.panY=0;applyZoom(); state.baseConfig=snapshot(); applySnapshot(state.baseConfig); updateHistoryButtons();
  }
  fileInput.addEventListener('change',()=>setFile(fileInput.files[0]));
  zone.addEventListener('dragover',e=>{e.preventDefault();zone.classList.add('drag')});
  zone.addEventListener('dragleave',()=>zone.classList.remove('drag'));
  zone.addEventListener('drop',e=>{e.preventDefault();zone.classList.remove('drag');const file=e.dataTransfer.files[0];if(!file)return;const dt=new DataTransfer();dt.items.add(file);fileInput.files=dt.files;setFile(file)});
  $('#remove').addEventListener('click',()=>clearWorkspace(true));
  function clearWorkspace(confirmReset) {
    if(confirmReset && state.history.length && !window.confirm('Đặt lại không gian làm việc và xóa ảnh cùng lịch sử xử lý hiện tại?')) return;
    state.requestId++;document.body.classList.remove('loading');fileInput.value='';state.file=null;state.history=[];state.index=-1;state.baseConfig=null;state.data=null;matchedReference=null;matchedDetails=null;$('#match-results').hidden=true;resetResults();
    if(state.objectUrl){URL.revokeObjectURL(state.objectUrl);state.objectUrl=null}
    $('#file-row').hidden=true;zone.style.display='flex';$('#viewer-stack').hidden=true;$('.empty').hidden=false;
    $('#image-status').textContent='Chọn ảnh để bắt đầu';$('#dim').textContent='—';$('#channels').textContent='—';$('#format').textContent='—';$('#filesize-meta').textContent='—';$('#submit').disabled=true;$('#run-match').disabled=true;errorBox.textContent='';updateHistoryButtons();
  }
  $('#reset').addEventListener('click',resetToOriginal);

  function restoreOriginalState() {
    state.requestId++;document.body.classList.remove('loading');$('#submit').disabled=!state.file;$('#submit').innerHTML='Phân tích ảnh <span>→</span>';
    state.data=null;resetResults();if(state.objectUrl){$('#viewer-original').src=state.objectUrl;$('#viewer-result').src=state.objectUrl;}
    $('#split').value=50;setSplit(50);$('#image-status').textContent='Ảnh gốc · chưa áp dụng xử lý';
    if(state.baseConfig)applySnapshot(state.baseConfig);state.range=[0,255];state.visible={};
    const title=state.file?.name||'';$('#filename').textContent=title;$('#dim').textContent='—';$('#channels').textContent='—';
  }
  function resetToOriginal() {
    if(!state.file)return;
    if(state.index>0&&!window.confirm('Đặt lại lịch sử xử lý và quay về ảnh gốc đã tải lên?'))return;
    state.requestId++;state.history=[null];state.index=0;restoreOriginalState();errorBox.textContent='';updateHistoryButtons();
  }

  function updateMethodControls() {
    const method=$('#method').value;
    $('#gamma-control').hidden=method!=='gamma';$('#clahe-control').hidden=method!=='clahe';$('#tile-control').hidden=method!=='clahe';
  }
  $('#method').addEventListener('change',updateMethodControls); updateMethodControls();
  form.addEventListener('input',updateHistoryButtons);form.addEventListener('change',updateHistoryButtons);

  async function process(config, saveHistory) {
    if(!state.file)return;
    const id=++state.requestId, button=$('#submit');button.disabled=true;button.textContent='Đang xử lý…';$('#image-status').textContent='Đang xử lý ảnh…';document.body.classList.add('loading');errorBox.textContent='';results.hidden=true;
    const body=new FormData();body.append('image',state.file,state.file.name);for(const [key,value] of Object.entries(config))body.append(key,value);
    try {
      const response=await fetch('/api/analyze',{method:'POST',body});
      let payload;try{payload=await response.json()}catch{throw new Error(`Máy chủ trả về lỗi HTTP ${response.status}. Hãy kiểm tra giới hạn tải lên.`)}
      if(id!==state.requestId)return;
      if(!response.ok||!payload.ok)throw new Error(payload.error?.message||`Phân tích thất bại (HTTP ${response.status}).`);
      state.data=payload.data;render(payload.data);
      if(saveHistory){const prev=state.history.slice(0,state.index+1);prev.push({...config});state.history=[prev[0]??null,...prev.slice(1).slice(-14)];state.index=state.history.length-1;updateHistoryButtons()}
      $('#image-status').textContent=`Phân tích hoàn tất · ${payload.data.info.width} × ${payload.data.info.height}`;
    } catch(error) {if(id===state.requestId){errorBox.textContent=error.message;$('#image-status').textContent='Không thể phân tích ảnh này';}}
    finally {if(id===state.requestId){button.disabled=false;button.innerHTML='Phân tích ảnh <span>→</span>';document.body.classList.remove('loading')}}
  }
  form.addEventListener('submit',e=>{e.preventDefault();process(snapshot(),true)});
  $('#undo').addEventListener('click',()=>{if(state.index>0){state.index--;const c=state.history[state.index];if(c===null)restoreOriginalState();else{applySnapshot(c);process(c,false)}updateHistoryButtons()}});
  $('#redo').addEventListener('click',()=>{if(state.index<state.history.length-1){state.index++;const c=state.history[state.index];if(c===null)restoreOriginalState();else{applySnapshot(c);process(c,false)}updateHistoryButtons()}});

  function render(d) {
    results.hidden=false;$('#warning').hidden=!d.warnings.length;$('#warning').textContent=d.warnings.join(' ');
    for(const option of $('#space').options)option.disabled=d.info.channels===1&&option.value!=='gray';$('#space').value='gray';
    $('#dim').textContent=`${d.info.width} × ${d.info.height}`;$('#channels').textContent=d.info.channels===1?'Ảnh xám':`${d.info.channels} kênh${d.info.alpha?' · alpha':''}`;
    $('#format').textContent=d.info.format;$('#filesize-meta').textContent=niceBytes(d.info.size);
    img('#viewer-original',d.original);img('#viewer-result',d.enhanced);$('#split').value=50;setSplit(50);
    img('#mask-original',d.mask_before);img('#mask-original-morph',d.mask_before_morph);img('#mask-enhanced',d.mask_after);img('#mask-enhanced-morph',d.mask_after_morph);
    $('#mask-stat-original').textContent=`Otsu: ${f(d.otsu_before.threshold)} · tiền cảnh ${f(d.mask_metrics_before.foreground_ratio*100)}% · hậu cảnh ${f(d.mask_metrics_before.background_ratio*100)}%`;
    $('#mask-stat-enhanced').textContent=`Otsu: ${f(d.otsu_after.threshold)} · tiền cảnh ${f(d.mask_metrics_after.foreground_ratio*100)}% · hậu cảnh ${f(d.mask_metrics_after.background_ratio*100)}%`;
    $('#threshold').textContent=f(d.otsu_after.threshold);$('#variance').textContent=f(d.otsu_after.between_class_variance);$('#fgbg').textContent=`${f(d.mask_metrics_after.foreground_ratio*100)}% / ${f(d.mask_metrics_after.background_ratio*100)}%`;
    const morph=d.morphology.operation==='none'?morphNames.none:`${morphNames[d.morphology.operation]}, kernel ${shapeNames[d.morphology.shape]} ${d.morphology.size}×${d.morphology.size}, ${d.morphology.iterations} lần lặp`;
    const modeName=d.mode==='binary_inv'?'nhị phân đảo':'nhị phân';
    $('#pipeline-detail').textContent=`${methodNames[d.method]} (${JSON.stringify(d.params)}) → Otsu ${modeName}, ngưỡng ${f(d.otsu_after.threshold)} → Hình thái học: ${morph}. Ảnh gốc vẫn được lưu.`;
    $('#morph-step').classList.toggle('active',d.morphology.operation!=='none');
    const variants=$('#variants');variants.replaceChildren();for(const [key,v] of Object.entries(d.variants)){
      const card=document.createElement('button');card.type='button';card.className='variant'+(key===d.method?' selected':'');card.setAttribute('aria-pressed',key===d.method?'true':'false');
      const preview=document.createElement('img');preview.src=v.preview;preview.alt=methodNames[key];
      const title=document.createElement('b');title.textContent=methodNames[key];const detail=document.createElement('small');detail.textContent=`${JSON.stringify(v.params)} · trung bình ${f(v.stats.mean)} · σ ${f(v.stats.std)}${key===d.method?' · đang dùng':''}`;
      card.append(preview,title,detail);card.addEventListener('click',()=>{$('#method').value=key;updateMethodControls();process(snapshot(),true)});variants.append(card);
    }
    state.visible={};$('#space').value='gray';state.range=[0,255];renderChannelToggles();renderStats();drawHistogram();renderComparison();renderExplorers(d);
  }
  function safeNumber(value){return value!==null&&value!==undefined&&Number.isFinite(Number(value))?Number(value):null}
  function renderComparison(){if(!state.data)return;const name=focusedChannel(),before=selectedSpaceData('before')?.channels[name],after=selectedSpaceData('after')?.channels[name];if(!before||!after)return;const metrics=[['Nhỏ nhất','min'],['Lớn nhất','max'],['Trung bình','mean'],['Trung vị','median'],['Độ lệch chuẩn','std'],['Dải động','dynamic_range'],['Entropy (bit)','entropy'],['Tương phản','contrast']];$('#comparison-rows').replaceChildren(...metrics.map(([label,key])=>{const a=safeNumber(before[key]),b=safeNumber(after[key]),delta=a===null||b===null?null:b-a,pct=a===null||b===null||a===0?null:(delta/a)*100;const tr=document.createElement('tr');[label,a===null?'—':f(a),b===null?'—':f(b),delta===null?'—':f(Math.abs(delta)),pct===null?(a===0&&delta!==null?'— (gốc = 0)':'—'):`${f(pct)}%`].forEach((value,i)=>{const cell=document.createElement(i===0?'th':'td');cell.textContent=value;tr.append(cell)});return tr}));const ob=state.data.otsu_before,oa=state.data.otsu_after;$('#threshold-before').textContent=f(ob.threshold);$('#threshold').textContent=f(oa.threshold);$('#variance-before').textContent=f(ob.between_class_variance);$('#variance').textContent=f(oa.between_class_variance);$('#fgbg-before').textContent=`${f(ob.foreground_ratio*100)}% / ${f(ob.background_ratio*100)}%`;$('#fgbg').textContent=`${f(oa.foreground_ratio*100)}% / ${f(oa.background_ratio*100)}%`}
  function drawSeries(canvasId,series,mode='line'){const el=$(`#${canvasId}`);if(!el)return;const box=el.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(260,box.width),h=210;el.width=Math.round(w*dpr);el.height=Math.round(h*dpr);const c=el.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);const pad={l:38,r:12,t:12,b:22},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;c.strokeStyle='#e9eaf0';c.fillStyle='#858a9c';c.font='9px DM Sans, sans-serif';for(let i=0;i<=4;i++){const y=pad.t+ph*i/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();c.fillText(mode==='cdf'?`${(1-i/4).toFixed(2)}`:'',3,y+3)}const max=mode==='cdf'?1:Math.max(1,...series.flatMap(s=>s.values.map(v=>Number(v)||0)));series.forEach(s=>{c.beginPath();c.strokeStyle=s.color;c.lineWidth=1.5;let has=false;s.values.forEach((raw,i)=>{const v=Number(raw)||0,x=pad.l+i/Math.max(1,s.values.length-1)*pw,y=pad.t+ph-(mode==='cdf'?v:v/max)*ph;if(!has){c.moveTo(x,y);has=true}else c.lineTo(x,y)});c.stroke()});c.fillStyle='#888';c.fillText('0',pad.l,h-5);c.fillText('255',w-pad.r-18,h-5);if(mode==='cdf'){c.fillText('CDF',pad.l+5,pad.t+10)}}
  function renderExplorers(d){
    const eq=d.equalization,box=$('#equalization-explorer');
    box.hidden=!eq;
    if(eq){
      const lutBody=$('#lut-row');
      lutBody.replaceChildren(...Array.from({length:256},(_,k)=>{
        const tr=document.createElement('tr');
        for(const v of [k,eq.histogram[k],eq.pdf[k].toFixed(5),eq.cdf[k].toFixed(5),eq.lut[k]]){
          const td=document.createElement('td');td.textContent=v;tr.append(td)
        }
        return tr
      }));
      $('#equalize-meta').textContent=eq.total_pixels.toLocaleString('vi-VN')+' pixel · Min '+eq.min+' · Max '+eq.max+(eq.constant?' · ảnh hằng, giữ nguyên':'');
      $('#equalize-cdfmin').textContent=eq.cdf_min.toFixed(6);
      const peaks=eq.histogram.map((count,level)=>({count,level})).filter(x=>x.count>0).sort((a,b)=>b.count-a.count).slice(0,5);
      $('#equalize-peaks').textContent='Mức xám có tần suất cao: '+(peaks.map(x=>x.level+' ('+x.count+')').join(' · ')||'không có');
      const lookup=()=>{const k=Math.min(255,Math.max(0,Number($('#lut-level').value)||0));$('#lut-result').textContent='Mức '+k+': '+eq.histogram[k]+' pixel · PDF '+eq.pdf[k].toFixed(5)+' · CDF '+eq.cdf[k].toFixed(5)+' · LUT → '+eq.lut[k]};
      $('#lut-level').oninput=lookup;lookup();
      drawSeries('equalize-pdf',[{values:eq.histogram,color:'#ef9a50'}]);
      drawSeries('equalize-cdf',[{values:eq.cdf,color:'#25896a'}],'cdf')
    }
    $('#otsu-explorer').hidden=false;
    img('#otsu-gray',d.grayscale_before);
    img('#otsu-binary',d.mode==='binary'?d.mask_after:d.mask_after_inverse);
    img('#otsu-inverse',d.mode==='binary'?d.mask_after_inverse:d.mask_after);
    $('#otsu-before-value').textContent=f(d.otsu_before.threshold);
    $('#otsu-after-value').textContent=f(d.otsu_after.threshold);
    $('#otsu-variance-delta').textContent=f(d.otsu_after.between_class_variance-d.otsu_before.between_class_variance);
    $('#otsu-class-ratios').textContent='Sau xử lý: tiền cảnh '+f(d.otsu_after.foreground_ratio*100)+'%, hậu cảnh '+f(d.otsu_after.background_ratio*100)+'%. Ảnh gốc: tiền cảnh '+f(d.otsu_before.foreground_ratio*100)+'%, hậu cảnh '+f(d.otsu_before.background_ratio*100)+'%. Chênh lệch: ngưỡng '+f(d.otsu_after.threshold-d.otsu_before.threshold)+', tiền cảnh '+f((d.otsu_after.foreground_ratio-d.otsu_before.foreground_ratio)*100)+' điểm %, hậu cảnh '+f((d.otsu_after.background_ratio-d.otsu_before.background_ratio)*100)+' điểm %.';
    const classes=(hist,t)=>{let n0=0,s0=0,n1=0,s1=0;for(let k=0;k<256;k++){const n=hist[k];if(k<=t){n0+=n;s0+=k*n}else{n1+=n;s1+=k*n}}return {w0:n0/(n0+n1),mu0:n0?s0/n0:null,w1:n1/(n0+n1),mu1:n1?s1/n1:null}};
    const h0=d.histograms_before.gray.channels.Intensity.histogram,h1=d.histograms_after.gray.channels.Intensity.histogram,c0=classes(h0,d.otsu_before.threshold),c1=classes(h1,d.otsu_after.threshold);
    const show=c=>'w₀='+f(c.w0)+' · μ₀='+(c.mu0===null?'—':f(c.mu0))+' · w₁='+f(c.w1)+' · μ₁='+(c.mu1===null?'—':f(c.mu1));
    $('#otsu-class-stats').textContent='Lớp mức thấp (≤ t) / cao (> t) — ảnh gốc: '+show(c0)+'; sau xử lý: '+show(c1)+'.';
    drawSeries('otsu-curve',[{values:d.otsu_before.variance_curve,color:'#ef9a50'},{values:d.otsu_after.variance_curve,color:'#6658d5'}])
  }
  function setSplit(value){$('#viewer-result').style.clipPath=`inset(0 ${100-value}% 0 0)`}
  $('#split').addEventListener('input',e=>setSplit(e.target.value));
  function applyZoom(){const stack=$('#viewer-stack');stack.style.transform=`translate(${state.panX}px,${state.panY}px) scale(${state.zoom})`;$('#zoom-level').textContent=`${Math.round(state.zoom*100)}%`}
  $('#zoom-in').addEventListener('click',()=>{state.zoom=Math.min(4,state.zoom*1.2);applyZoom()});$('#zoom-out').addEventListener('click',()=>{state.zoom=Math.max(.5,state.zoom/1.2);applyZoom()});
  $('#fullscreen').addEventListener('click',async()=>{try{if(!document.fullscreenElement)await $('#viewer').requestFullscreen();else await document.exitFullscreen()}catch{errorBox.textContent='Trình duyệt không hỗ trợ chế độ toàn màn hình.'}});
  $('#zoom-level').addEventListener('click',()=>{state.zoom=1;state.panX=0;state.panY=0;applyZoom()});$('#fit').addEventListener('click',()=>{state.zoom=1;state.panX=0;state.panY=0;applyZoom()});
  $('#viewer').addEventListener('pointerdown',e=>{if(e.target=== $('#split'))return;state.drag={x:e.clientX,y:e.clientY,px:state.panX,py:state.panY};$('#viewer').setPointerCapture(e.pointerId)});
  $('#viewer').addEventListener('pointermove',e=>{if(state.drag){state.panX=state.drag.px+e.clientX-state.drag.x;state.panY=state.drag.py+e.clientY-state.drag.y;applyZoom()}});
  for(const ev of ['pointerup','pointercancel'])$('#viewer').addEventListener(ev,()=>state.drag=null);

  function selectedSpaceData(side){return state.data?.[`histograms_${side}`]?.[$('#space').value]}
  function channelNames(){const a=selectedSpaceData('before');return a?Object.keys(a.channels):[]}
  function renderChannelToggles(){const box=$('#channel-toggles');box.replaceChildren();const names=channelNames();names.forEach((name,i)=>{if(state.visible[name]===undefined)state.visible[name]=true;const b=document.createElement('button');b.className='channel-toggle'+(state.visible[name]?'':' off');b.textContent=name;b.setAttribute('aria-pressed',state.visible[name]);b.addEventListener('click',()=>{state.visible[name]=!state.visible[name];b.classList.toggle('off',!state.visible[name]);b.setAttribute('aria-pressed',state.visible[name]);renderStats();renderComparison();drawHistogram()});box.append(b)})}
  $('#space').addEventListener('change',()=>{state.range=[0,255];state.visible={};renderChannelToggles();renderStats();renderComparison();drawHistogram()});
  function focusedChannel(){const names=channelNames();return names.find(n=>state.visible[n])||names[0]}
  function renderStats(){if(!state.data)return;const name=focusedChannel(),after=selectedSpaceData('after')?.channels[name];if(!after)return;$('#stats-label').textContent=`${$('#space').selectedOptions[0].text} · kênh ${name} · 0–${after.value_max}`}
  const seriesColors={R:'#df5861',G:'#299b70',B:'#517bd8',Intensity:'#6658d5',H:'#d85db5',S:'#e69f32',V:'#517bd8',L:'#6658d5',a:'#da625c',b:'#cf9d28',Y:'#6658d5',Cr:'#df5861',Cb:'#517bd8'};
  function drawHistogram(){if(!state.data)return;const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height,left=45,right=12,top=16,bottom=29,plotW=w-left-right,plotH=h-top-bottom;
    ctx.clearRect(0,0,w,h);ctx.font='9px DM Sans, sans-serif';ctx.strokeStyle='#ececf2';ctx.fillStyle='#8c8f9d';ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const y=top+plotH*i/4;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();ctx.textAlign='right';ctx.fillText(i===4?'0':`${Math.round(maxCount()*(1-i/4))}`,left-6,y+3)}
    const start=state.range[0],end=state.range[1],xFor=x=>left+(x-start)/(end-start)*plotW;if($('#space').value==='gray'){const t=state.data.otsu_after.threshold,tx=xFor(t);if(tx>left&&tx<w-right){ctx.fillStyle='rgba(102,88,213,.045)';ctx.fillRect(left,top,tx-left,plotH);ctx.fillStyle='rgba(37,137,106,.045)';ctx.fillRect(tx,top,w-right-tx,plotH);ctx.fillStyle='#777';ctx.font='9px DM Sans, sans-serif';ctx.fillText('≤ t',left+4,top+11);ctx.fillText('> t',Math.min(w-right-22,tx+4),top+11)}}
    for(let x=Math.ceil(start/32)*32;x<=end;x+=32){const px=xFor(x);ctx.strokeStyle='#f1f1f5';ctx.beginPath();ctx.moveTo(px,top);ctx.lineTo(px,top+plotH);ctx.stroke();ctx.fillStyle='#898c99';ctx.textAlign='center';ctx.fillText(String(x),px,h-9)}
    const names=channelNames();for(const name of names){if(!state.visible[name])continue;for(const side of ['before','after']){const row=selectedSpaceData(side).channels[name],arr=row.histogram,max=maxCount();ctx.beginPath();let started=false;for(let val=Math.ceil(start);val<=end;val++){const x=xFor(val),y=top+plotH-(arr[val]/max)*plotH;if(!started){ctx.moveTo(x,y);started=true}else ctx.lineTo(x,y)}ctx.strokeStyle=seriesColors[name]||'#6658d5';ctx.globalAlpha=side==='before'?.46:1;ctx.lineWidth=side==='before'?1:1.5;ctx.setLineDash(side==='before'?[3,2]:[]);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1}}
    const selected=focusedChannel(),metric=selectedSpaceData('after')?.channels[selected];if(metric){for(const [val,color,dash] of [[metric.min,'#888', [2,3]],[metric.max,'#888',[2,3]],[metric.mean,'#ba8b26',[5,3]],[metric.median,'#299b70',[1,3]]]){if(val<start||val>end)continue;ctx.beginPath();ctx.moveTo(xFor(val),top);ctx.lineTo(xFor(val),top+plotH);ctx.strokeStyle=color;ctx.lineWidth=1;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([])}}
    if($('#space').value==='gray'){const threshold=state.data.otsu_after.threshold;if(threshold>=start&&threshold<=end){ctx.beginPath();ctx.moveTo(xFor(threshold),top);ctx.lineTo(xFor(threshold),top+plotH);ctx.strokeStyle='#262738';ctx.lineWidth=1.2;ctx.setLineDash([5,3]);ctx.stroke();ctx.setLineDash([])}}
    canvas.dataset.plot=JSON.stringify({left,right,top,bottom,plotW,plotH,start,end,max:maxCount()});
  }
  function maxCount(){let max=1;for(const side of ['before','after'])for(const [name,row] of Object.entries(selectedSpaceData(side)?.channels||{}))if(state.visible[name]!==false)for(const count of row.histogram)max=Math.max(max,count);return max}
  canvas.addEventListener('wheel',e=>{if(!state.data)return;e.preventDefault();const width=state.range[1]-state.range[0],factor=e.deltaY<0?.8:1.25,newWidth=Math.max(16,Math.min(256,width*factor)),center=(state.range[0]+state.range[1])/2;state.range=[Math.max(0,center-newWidth/2),Math.min(255,center+newWidth/2)];drawHistogram()},{passive:false});
  canvas.addEventListener('pointerdown',e=>{if(!state.data)return;state.drag={x:e.clientX,range:[...state.range],chart:true};canvas.setPointerCapture(e.pointerId)});
  canvas.addEventListener('pointermove',e=>{const rect=canvas.getBoundingClientRect(),p=JSON.parse(canvas.dataset.plot||'{}');if(state.drag?.chart){const delta=(e.clientX-state.drag.x)/p.plotW*(state.drag.range[1]-state.drag.range[0]);let start=state.drag.range[0]-delta,end=state.drag.range[1]-delta;if(start<0){end-=start;start=0}if(end>255){start-=end-255;end=255}state.range=[Math.max(0,start),Math.min(255,end)];drawHistogram();return}if(!p.plotW)return;const x=e.clientX-rect.left;if(x<p.left||x>rect.width-p.right){tooltip.hidden=true;return}const val=Math.round(p.start+(x-p.left)/p.plotW*(p.end-p.start)),name=focusedChannel(),a=selectedSpaceData('before')?.channels[name]?.histogram[val],b=selectedSpaceData('after')?.channels[name]?.histogram[val];tooltip.hidden=false;tooltip.style.left=`${Math.min(rect.width-130,Math.max(0,x+8))}px`;tooltip.style.top='8px';tooltip.textContent=`${name} · mức ${val} · ảnh gốc ${a} Pixel · ảnh tăng cường ${b} Pixel`});
  for(const ev of ['pointerup','pointercancel'])canvas.addEventListener(ev,()=>{if(state.drag?.chart)state.drag=null});$('#zoom-reset').addEventListener('click',()=>{state.range=[0,255];drawHistogram()});
  $('#equalization-explorer').addEventListener('toggle',()=>{const eq=state.data?.equalization;if($('#equalization-explorer').open&&eq){drawSeries('equalize-pdf',[{values:eq.histogram,color:'#ef9a50'}]);drawSeries('equalize-cdf',[{values:eq.cdf,color:'#25896a'}],'cdf')}});
  window.addEventListener('resize',()=>{drawHistogram();if(state.data)drawSeries('otsu-curve',[{values:state.data.otsu_before.variance_curve,color:'#ef9a50'},{values:state.data.otsu_after.variance_curve,color:'#6658d5'}]);const eq=state.data?.equalization;if(eq&&$('#equalization-explorer').open){drawSeries('equalize-pdf',[{values:eq.histogram,color:'#ef9a50'}]);drawSeries('equalize-cdf',[{values:eq.cdf,color:'#25896a'}],'cdf')}if(matchedDetails&&!$('#match-results').hidden){drawSeries('match-histogram',[{values:matchedDetails.source_histogram,color:'#ef9a50'},{values:matchedDetails.reference_histogram,color:'#25896a'},{values:matchedDetails.matched_histogram,color:'#6658d5'}]);drawSeries('match-cdf',[{values:matchedDetails.source_cdf,color:'#ef9a50'},{values:matchedDetails.reference_cdf,color:'#25896a'},{values:matchedDetails.matched_cdf,color:'#6658d5'}],'cdf')}});

  $('#export-chart').addEventListener('click',async()=>{if(!state.data)return;try{const space=$('#space').value,channels=channelNames().filter(name=>state.visible[name]),histogram=state.data.histograms_after[space],before=state.data.histograms_before[space];if(!channels.length)throw new Error('Bật ít nhất một kênh trước khi xuất biểu đồ.');const response=await fetch('/api/export/histogram',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({histogram,before,space,channels,range:state.range,threshold:space==='gray'?state.data.otsu_after.threshold:null})});const payload=await response.json();if(!response.ok||!payload.ok)throw new Error(payload.error?.message||'Không thể xuất biểu đồ.');const blob=await fetch(payload.data.image).then(r=>r.blob());saveBlob(blob,`histogram-${space}-${channels.join('-')}.png`)}catch(error){errorBox.textContent=error.message}});
  $('#download-image').addEventListener('click',e=>{e.preventDefault();downloadAsset('enhanced')});
  $('#download-mask').addEventListener('click',e=>{e.preventDefault();downloadAsset('mask_after_morph')});
  $('#reference-file').addEventListener('change',()=>{matchedReference=null;matchedDetails=null;$('#match-results').hidden=true;$('#run-match').disabled=!(state.file&&$('#reference-file').files[0]);$('#match-status').textContent=$('#reference-file').files[0]?'Sẵn sàng chạy Histogram Matching.':'Chọn ảnh tham chiếu.'});
  $('#run-match').addEventListener('click',async()=>{const reference=$('#reference-file').files[0];if(!state.file||!reference)return;const button=$('#run-match');button.disabled=true;$('#match-results').hidden=true;$('#match-status').textContent='Đang khớp Histogram…';try{const fd=new FormData();fd.append('image',state.file,state.file.name);fd.append('reference',reference,reference.name);const response=await fetch('/api/match',{method:'POST',body:fd});const payload=await response.json();if(!response.ok||!payload.ok)throw new Error(payload.error?.message||`Histogram Matching thất bại (HTTP ${response.status}).`);matchedReference=reference;const d=payload.data,stats=d.details;matchedDetails=stats;img('#match-source',d.source);img('#match-reference',d.reference);img('#match-output',d.matched);$('#match-results').hidden=false;$('#match-distance').textContent=`Độ lệch CDF trung bình (trước → sau): ${f(stats.mean_cdf_distance_before)} → ${f(stats.mean_cdf_distance_after)}`;$('#match-metric-note').textContent=`${stats.channel}. ${stats.metric_note}`;drawSeries('match-histogram',[{values:stats.source_histogram,color:'#ef9a50'},{values:stats.reference_histogram,color:'#25896a'},{values:stats.matched_histogram,color:'#6658d5'}]);drawSeries('match-cdf',[{values:stats.source_cdf,color:'#ef9a50'},{values:stats.reference_cdf,color:'#25896a'},{values:stats.matched_cdf,color:'#6658d5'}],'cdf');$('#match-status').textContent='Đã hoàn tất; lịch sử xử lý ảnh nguồn không bị thay đổi.'}catch(error){$('#match-status').textContent=error.message}finally{button.disabled=!(state.file&&$('#reference-file').files[0])}});
  $('#download-match').addEventListener('click',async()=>{if(!state.file||!matchedReference)return;const fd=new FormData();fd.append('image',state.file,state.file.name);fd.append('reference',matchedReference,matchedReference.name);try{const response=await fetch('/api/export/match',{method:'POST',body:fd});if(!response.ok){const payload=await response.json().catch(()=>({}));throw new Error(payload.error?.message||`Xuất ảnh thất bại (HTTP ${response.status}).`)}saveBlob(await response.blob(),'histogram-matched.png')}catch(error){$('#match-status').textContent=error.message}});
  async function downloadAsset(asset){if(!state.file)return;const formData=new FormData();formData.append('image',state.file,state.file.name);for(const [key,value] of Object.entries(snapshot()))formData.append(key,value);formData.append('asset',asset);try{const response=await fetch('/api/export/image',{method:'POST',body:formData});if(!response.ok){const payload=await response.json().catch(()=>({}));throw new Error(payload.error?.message||`Xuất ảnh thất bại (HTTP ${response.status}).`)}const blob=await response.blob();const disposition=response.headers.get('Content-Disposition')||'';const filename=disposition.match(/filename="?([^";]+)"?/i)?.[1]||`${asset}.png`;saveBlob(blob,filename)}catch(error){errorBox.textContent=error.message}}
  $('#export-csv').addEventListener('click',()=>{if(!state.data)return;const space=$('#space').value,rows=[['Giai đoạn','Không gian màu','Kênh','Nhỏ nhất','Lớn nhất','Trung bình','Trung vị','Độ lệch chuẩn','Dải động','Entropy (bit)','Tương phản','Mức','Số Pixel']];for(const side of ['before','after'])for(const [channel,s] of Object.entries(state.data[`histograms_${side}`][space].channels))for(let bin=0;bin<256;bin++)rows.push([side,space,channel,s.min,s.max,s.mean,s.median,s.std,s.dynamic_range,s.entropy,s.contrast,bin,s.histogram[bin]]);const csv=rows.map(row=>row.join(',')).join('\r\n');saveBlob(new Blob([csv],{type:'text/csv;charset=utf-8'}),'phan-tich-anh.csv')});
  function saveBlob(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
})();
