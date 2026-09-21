const $ = (id) => document.getElementById(id);
const names = {linear:'TF-IDF + 逻辑回归',cnn_plain:'TextCNN',cnn_weighted:'TextCNN · 类别加权'};
const labels = ['消极','中性','积极'];
const samples = ['业绩增长不错，继续看好后市。','补仓补得心力憔悴，还是一直跌。','今天先观望，等公告出来再决定。'];
let errors = [], page = 0;
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function percentage(x){return (x*100).toFixed(1)+'%';}
function updateCount(){$('charcount').textContent=$('text').value.length+' / 2000';}
$('text').addEventListener('input',updateCount);
$('sample').addEventListener('change',()=>{if($('sample').value!==''){$('text').value=samples[Number($('sample').value)];updateCount();}});
$('clear').addEventListener('click',()=>{$('text').value='';$('sample').value='';$('predictions').replaceChildren(document.createTextNode('尚无分析结果'));$('predictions').className='empty';$('message').textContent='';updateCount();});
document.querySelectorAll('.tab').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-selected',String(b===button));});document.querySelectorAll('.view').forEach(v=>v.hidden=v.id!==button.dataset.view);}));
$('submit').addEventListener('click',async()=>{
  const text=$('text').value;if(!text.trim()){$('message').textContent='请输入待分析文本';return;}
  $('submit').disabled=true;$('message').textContent='正在分析';
  try{const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});const data=await response.json();if(!response.ok)throw new Error(data.error);
    const target=$('predictions');target.className='';target.replaceChildren();
    Object.entries(data.models).forEach(([key,value])=>{const item=node('div',undefined,'prediction');const head=node('div',undefined,'prediction-head');head.append(node('span',names[key],'model-name'));const verdict=node('span',value.label,'verdict');if(value.review)verdict.append(node('span','待复核','review'));head.append(verdict);item.append(head);value.scores.forEach((score,i)=>{const row=node('div',undefined,'bar-row');const track=node('div',undefined,'track');const fill=node('div',undefined,'fill');fill.style.width=percentage(score);track.append(fill);row.append(node('span',labels[i]),track,node('span',percentage(score),'score'));item.append(row);});target.append(item);});
    $('message').textContent=data.truncated?'已使用规范化后的前128字':'分析完成';
  }catch(error){$('message').textContent=error.message||'分析失败';}finally{$('submit').disabled=false;}
});
function renderErrors(){const label=$('labelFilter').value,model=$('modelFilter').value,query=$('query').value.trim();const filtered=errors.filter(r=>(!label||r.label===label)&&(!model||r.models[model].label!==r.label)&&(!query||r.text.includes(query)));const total=Math.max(1,Math.ceil(filtered.length/20));page=Math.min(page,total-1);const tbody=$('errorRows');tbody.replaceChildren();filtered.slice(page*20,(page+1)*20).forEach(r=>{const tr=node('tr');tr.append(node('td',r.text),node('td',r.label));Object.keys(names).forEach(n=>tr.append(node('td',r.models[n].label+' · '+percentage(r.models[n].score),r.models[n].label!==r.label?'bad':'')));tbody.append(tr);});if(!filtered.length){const td=node('td','没有匹配的记录');td.colSpan=5;const tr=node('tr');tr.append(td);tbody.append(tr);}$('errorCount').textContent=filtered.length+' 条';$('pageInfo').textContent=(page+1)+' / '+total;$('prev').disabled=page===0;$('next').disabled=page>=total-1;}
['labelFilter','modelFilter','query'].forEach(id=>$(id).addEventListener('input',()=>{page=0;renderErrors();}));$('prev').addEventListener('click',()=>{page--;renderErrors();});$('next').addEventListener('click',()=>{page++;renderErrors();});
async function init(){try{const [report,errorData]=await Promise.all([fetch('/api/report').then(r=>{if(!r.ok)throw Error();return r.json();}),fetch('/api/errors').then(r=>r.json())]);errors=errorData;const audit=report.audit;$('trainCount').textContent=audit.split_counts.train.toLocaleString();$('valCount').textContent=audit.split_counts.validation.toLocaleString();$('testCount').textContent=audit.split_counts.test.toLocaleString();Object.entries(report.models).forEach(([key,value])=>{const tr=node('tr');[names[key],value.validation_macro_f1.toFixed(4),percentage(value.test.accuracy),value.test.macro_f1.toFixed(4)].forEach(t=>tr.append(node('td',t)));$('metrics').append(tr);});$('protocol').textContent='模型选择依据：验证集 Macro-F1。当前选中 '+names[report.selected_model]+'。测试集不参与选择。';$('audit').textContent='原始训练 / 评估文件：'+audit.raw_train+' / '+audit.raw_eval+' 条。去除重复记录 '+audit.duplicate_rows_removed+' 条；剔除标签冲突文本 '+audit.conflicting_texts_removed+' 组。清洗后三个划分的重复输入为 0。';$('connection').textContent='本地模型就绪';renderErrors();}catch(error){$('connection').textContent='连接失败';$('message').textContent='无法加载实验结果';}}
init();
