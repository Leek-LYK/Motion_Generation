"""Standalone offline reviewer page; receives only anonymous public trial data."""
import json


def review_html(data):
    encoded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 动作盲评</title>
<style>
body{font:16px system-ui,sans-serif;background:#101722;color:#ebf1f7;max-width:1400px;margin:24px auto;padding:0 18px}
h1{font-size:24px}button,input,select,textarea{font:inherit;margin:5px;padding:7px;border-radius:5px;border:1px solid #728499}
button{cursor:pointer}video{width:100%;background:#000}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
article,fieldset{border:1px solid #42566e;padding:12px;margin:12px 0;border-radius:8px}p{line-height:1.6}textarea{width:95%}
.muted{color:#adc0d5}.badge{background:#493722;padding:8px}.scores{display:flex;flex-wrap:wrap;gap:10px}
@media(max-width:850px){.pair{grid-template-columns:1fr}}label{display:inline-block}
</style>
<h1>G1 动作盲评</h1><p id="scope" class="badge"></p>
<p>根据文本分别判断两段视频的主动作，并选择更符合指令、动作更完整自然的一段。可选平局。适用跳跃还需判断是否能辨认完整的起跳、腾空与落地。方向、速度、停步和手臂等问题请记入备注。</p>
<fieldset><legend>评阅者</legend><label>评分文件槽 <select id="slot"><option value="1">1</option><option value="2">2</option></select></label>
<label>评阅者 ID <input id="reviewer" autocomplete="off"></label><p class="muted">两位独立评阅者分别填写，不共享或协商评分。选择文件槽后可在当前浏览器保存草稿。</p></fieldset>
<nav><button id="previous">上一条</button><select id="navigation"></select><button id="next">下一条</button><span id="progress"></span></nav>
<article><p id="prompt"></p><p class="muted" id="trial"></p>
<button id="play">同时播放</button><button id="pause">同时暂停</button><button id="restart">回到开头</button>
<div class="pair"><section><h2>A</h2><video id="A" controls preload="metadata"></video><div class="scores">
<label>主动作 <select data-field="A_main_correct"><option value="">请选择</option><option value="1">正确</option><option value="0">错误</option></select></label>
<label>跳跃完整 <select data-field="A_jump_complete"><option value="">请选择</option><option value="1">完整</option><option value="0">不完整</option></select></label></div></section>
<section><h2>B</h2><video id="B" controls preload="metadata"></video><div class="scores">
<label>主动作 <select data-field="B_main_correct"><option value="">请选择</option><option value="1">正确</option><option value="0">错误</option></select></label>
<label>跳跃完整 <select data-field="B_jump_complete"><option value="">请选择</option><option value="1">完整</option><option value="0">不完整</option></select></label></div></section></div>
<p id="jumpNote" class="muted"></p><label>总体偏好 <select data-field="preference"><option value="">请选择</option><option value="A">A</option><option value="B">B</option><option value="tie">平局</option></select></label>
<p><textarea data-field="direction_speed_arms_notes" rows="3" placeholder="动作、方向、速度、手臂、停步、接触等具体问题"></textarea></p></article>
<fieldset><legend>评阅声明</legend><label><input type="checkbox" id="human">我是人工评阅者</label>
<label><input type="checkbox" id="independent_review">我已独立评阅</label><label><input type="checkbox" id="model_mapping_not_viewed">评分期间未查看模型映射</label>
<p>这三项来自评分统计协议；默认未勾选。空白记录与未完成声明会保留为待完成。</p></fieldset>
<button id="exportCSV">导出评分 CSV</button><button id="exportAttestation">导出评阅声明 JSON</button><p id="message" class="muted"></p>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
'use strict';
const packet=JSON.parse(document.getElementById('data').textContent), trials=packet.trials;
const fields=['preference','A_main_correct','B_main_correct','A_jump_complete','B_jump_complete','direction_speed_arms_notes'];
const byId=id=>document.getElementById(id), controls=[...document.querySelectorAll('[data-field]')];
let position=0, draft={reviewer:'',rows:{},attestation:{}};
const key=()=>`g1-review:${packet.package_id}:${byId('slot').value}`;
function save(){draft.reviewer=byId('reviewer').value;for(const k of ['human','independent_review','model_mapping_not_viewed'])draft.attestation[k]=byId(k).checked;try{localStorage.setItem(key(),JSON.stringify(draft))}catch(e){byId('message').textContent='浏览器未提供本地存储，请导出文件保存草稿。'}}
function load(){try{draft=JSON.parse(localStorage.getItem(key())||'null')||{reviewer:'',rows:{},attestation:{}}}catch(e){draft={reviewer:'',rows:{},attestation:{}}}byId('reviewer').value=draft.reviewer;for(const k of ['human','independent_review','model_mapping_not_viewed'])byId(k).checked=draft.attestation[k]===true;show();}
function complete(t){const r=draft.rows[t.trial_id]||{};return ['A','B','tie'].includes(r.preference)&&['A_main_correct','B_main_correct'].every(k=>['0','1'].includes(r[k]))&&(!t.complete_jump_applicable||['A_jump_complete','B_jump_complete'].every(k=>['0','1'].includes(r[k])))}
function progress(){byId('progress').textContent=`已完成 ${trials.filter(complete).length} / ${trials.length}`;}
function show(){const t=trials[position], row=draft.rows[t.trial_id]||{};byId('prompt').textContent=t.prompt;byId('trial').textContent=`试次 ${position+1} / ${trials.length} · ${t.trial_id}`;byId('navigation').value=String(position);for(const side of ['A','B']){byId(side).pause();byId(side).src=t['video_'+side];}for(const el of controls){el.value=row[el.dataset.field]||'';el.disabled=el.dataset.field.endsWith('jump_complete')&&!t.complete_jump_applicable;}byId('jumpNote').textContent=t.complete_jump_applicable?'本条需要填写两段视频的完整跳跃判断。':'本条不要求完整跳跃判断。';progress();}
function download(name,type,text){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csvCell(v){return '"'+String(v??'').replaceAll('"','""')+'"'}
for(let i=0;i<trials.length;i++){const option=document.createElement('option');option.value=String(i);option.textContent=`第 ${i+1} 条`;byId('navigation').appendChild(option)}
for(const el of controls)el.addEventListener('input',()=>{const id=trials[position].trial_id;draft.rows[id]??={};draft.rows[id][el.dataset.field]=el.value;save();progress();});
byId('slot').addEventListener('change',load);byId('reviewer').addEventListener('input',save);
for(const k of ['human','independent_review','model_mapping_not_viewed'])byId(k).addEventListener('change',save);
byId('navigation').addEventListener('change',()=>{position=Number(byId('navigation').value);show()});
byId('previous').onclick=()=>{position=Math.max(0,position-1);show()};byId('next').onclick=()=>{position=Math.min(trials.length-1,position+1);show()};
byId('play').onclick=()=>{for(const side of ['A','B'])byId(side).play().catch(()=>{})};byId('pause').onclick=()=>{for(const side of ['A','B'])byId(side).pause()};byId('restart').onclick=()=>{for(const side of ['A','B']){byId(side).pause();byId(side).currentTime=0;}};
byId('exportCSV').onclick=()=>{save();const header=['trial_id','reviewer_id',...fields],rows=trials.map(t=>{const r=draft.rows[t.trial_id]||{};return [t.trial_id,draft.reviewer,...fields.map(k=>r[k]||'')].map(csvCell).join(',')});download(`rater_${byId('slot').value}.csv`,'text/csv;charset=utf-8','\uFEFF'+header.map(csvCell).join(',')+'\r\n'+rows.join('\r\n')+'\r\n');byId('message').textContent='已导出当前评分。请另行导出评阅声明；草稿可以保留空白。';};
byId('exportAttestation').onclick=()=>{save();download(`rater_${byId('slot').value}_attestation.json`,'application/json',JSON.stringify({reviewer_id:draft.reviewer,...draft.attestation},null,2))};
byId('scope').textContent=packet.engineering_fixture?'开发流程试运行：本包不用于 M5 最终人工验收。':'请独立完成全部预注册试次。';load();
</script></html>'''.replace('__DATA_JSON__', encoded, 1)
