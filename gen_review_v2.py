#!/usr/bin/env python3
"""Generate clean review HTML - v2 with working buttons."""
import json

with open("tmp_review_results.json", encoding="utf-8") as f:
    results = json.load(f)

data_json = json.dumps(results, ensure_ascii=False)

# Build JS data for each card
card_data = []
for item in results:
    ev = item.get("semantic_evidence", [])
    ev_labels = json.dumps([{"label": e["label"], "label_zh": e["label_zh"]} for e in ev], ensure_ascii=False)
    card_data.append({
        "id": item["id"],
        "speaker": item["speaker"],
        "emotion": item["emotion"],
        "duration": item["duration"],
        "sds": item["sds"],
        "gt_text": item["gt_text"],
        "whisper_text": item["whisper_text"] or "",
        "speech_rate": item.get("speech_rate", 0),
        "pause_ratio": item.get("pause_ratio", 0),
        "evidence_count": item.get("evidence_count", 0),
        "safety_flag": item.get("safety_flag", False),
        "evidence": ev
    })

cards_json = json.dumps(card_data, ensure_ascii=False)

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>语音模块 人工核对 — 50段音频评审</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,'Microsoft YaHei',sans-serif;background:#f5f5f5;padding:20px;max-width:900px;margin:0 auto}
h1{margin-bottom:10px}
.progress{margin:15px 0}
.progress-bar{height:8px;background:#e0e0e0;border-radius:4px;overflow:hidden}
.progress-fill{height:100%;background:#4CAF50;width:0%;transition:width .3s}
.stats{display:flex;gap:15px;margin:10px 0 15px;flex-wrap:wrap}
.stats div{background:#fff;padding:6px 14px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.card{background:#fff;border-radius:8px;margin-bottom:14px;padding:14px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.card-hdr{display:flex;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap}
.card-id{font-weight:bold;font-size:15px}
.card-meta{color:#666;font-size:12px}
audio{width:100%;max-width:400px}
.txt{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0}
@media(max-width:700px){.txt{grid-template-columns:1fr}}
.tb{padding:8px 10px;border-radius:6px;font-size:13px;line-height:1.6}
.gt{background:#e8f5e9;border-left:3px solid #4CAF50}
.wh{background:#fff3e0;border-left:3px solid #FF9800}
.tl{font-weight:bold;font-size:11px;margin-bottom:3px;color:#666}
.ev{margin:6px 0}
.ev-tag{display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px;margin:2px}
.ev-sleep{background:#e3f2fd;color:#1565c0}
.ev-lonely{background:#fce4ec;color:#c62828}
.ev-anxiety{background:#fff3e0;color:#e65100}
.ev-interest{background:#f3e5f5;color:#6a1b9a}
.ev-repeat{background:#e8eaf6;color:#283593}
.ev-time{background:#efebe9;color:#4e342e}
.rats{display:flex;gap:16px;margin:8px 0;flex-wrap:wrap}
.rg{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.rg label{font-size:12px;font-weight:bold;min-width:70px}
.btn{padding:3px 10px;border:1px solid #ddd;border-radius:4px;cursor:pointer;font-size:12px;background:#fafafa}
.btn:hover{background:#eee}
.sg{background:#4CAF50;color:#fff;border-color:#4CAF50}
.so{background:#FFC107;color:#fff;border-color:#FFC107}
.sb{background:#f44336;color:#fff;border-color:#f44336}
.ni{width:100%;margin-top:6px;padding:5px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px}
.act{position:sticky;bottom:0;background:#fff;padding:10px 16px;box-shadow:0 -2px 10px rgba(0,0,0,.1);display:flex;gap:10px;align-items:center;border-radius:8px 8px 0 0;flex-wrap:wrap}
.act button{padding:7px 16px;border:none;border-radius:6px;cursor:pointer;font-size:13px}
.bs{background:#2196F3;color:#fff}
.be{background:#4CAF50;color:#fff}
.sf{color:#f44336;font-weight:bold;font-size:12px}
</style>
</head>
<body>
<h1>&#127928; 语音模块转写与语义标签 人工核对</h1>
<p>共50段音频 &middot; 听音频 &rarr; 对比GT和Whisper &rarr; 打分</p>

<div class="stats" id="stats"></div>
<div class="progress"><div class="progress-bar"><div class="progress-fill" id="pf"></div></div></div>
<div id="cards"></div>
<div class="act">
  <button class="bs" id="btnSave">&#128190; 保存进度</button>
  <button class="be" id="btnExport">&#128228; 导出结果</button>
  <span style="color:#666;font-size:12px" id="sv"></span>
</div>

<script>
const CARDS = """ + cards_json + """;

// ----------渲染引擎：直接用 DOM API，不用 innerHTML 模板拼接 ----------
let R = JSON.parse(localStorage.getItem("r") || "{}");
let N = JSON.parse(localStorage.getItem("n") || "{}");

const CLASSES = {good:"sg", ok:"so", bad:"sb"};
const EMOJI = {good:"&#9989;", ok:"&#9888;&#65039;", bad:"&#10060;"};
const LABELS = {good:"正确", ok:"部分错", bad:"错误"};
const EV_CLASS = {sleep_complaint:"ev-sleep", loneliness:"ev-lonely", anxiety_worry:"ev-anxiety", loss_of_interest:"ev-interest", repeated_questions:"ev-repeat", time_confusion:"ev-time"};
const EV_NAME = {sleep_complaint:"睡眠抱怨", loneliness:"孤独表达", anxiety_worry:"焦虑担忧", loss_of_interest:"兴趣下降", repeated_questions:"重复问题", time_confusion:"时间混乱"};

function esc(s){if(!s)return "";return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}

function render(){
  const c = document.getElementById("cards");
  c.innerHTML = "";
  let tc=0, lc=0;
  CARDS.forEach(function(item){
    const id = item.id;
    const r = R[id] || {};
    if(r.t) tc++;
    if(r.l) lc++;
    const idStr = String(id).padStart(2,"0");
    const card = document.createElement("div");
    card.className = "card";

    // Header
    const hdr = document.createElement("div");
    hdr.className = "card-hdr";
    hdr.innerHTML = '<div><span class="card-id">#'+id+'</span> <span class="card-meta">'+esc(item.speaker)+' / '+esc(item.emotion)+' / '+item.duration+'s / SDS='+(item.sds ?? "N/A")+'</span></div>'
      + '<div><span class="card-meta">语速:'+(item.speech_rate?item.speech_rate.toFixed(1):"?")+'&nbsp; 停顿:'+(item.pause_ratio?Math.round(item.pause_ratio*100):"?")+'%</span></div>';
    card.appendChild(hdr);

    // Audio player
    const ap = document.createElement("div");
    ap.innerHTML = '<audio controls src="audio/'+idStr+'_'+item.speaker+'_'+item.emotion+'.wav"></audio>';
    card.appendChild(ap);

    // Text comparison
    const txt = document.createElement("div");
    txt.className = "txt";
    txt.innerHTML = '<div class="tb gt"><div class="tl">GT 原文</div>'+esc(item.gt_text)+'</div>'
      + '<div class="tb wh"><div class="tl">Whisper 转写</div>'+(item.whisper_text?esc(item.whisper_text):'<i style="color:#999">(空)</i>')+'</div>';
    card.appendChild(txt);

    // Evidence
    if(item.evidence && item.evidence.length > 0){
      const ev = document.createElement("div");
      ev.className = "ev";
      item.evidence.forEach(function(e){
        const span = document.createElement("span");
        span.className = "ev-tag "+(EV_CLASS[e.label]||"");
        span.textContent = EV_NAME[e.label] || e.label_zh;
        ev.appendChild(span);
      });
      card.appendChild(ev);
    }
    if(item.safety_flag){
      const sf = document.createElement("div");
      sf.className = "sf";
      sf.textContent = "⚠️ 敏感词触发";
      card.appendChild(sf);
    }

    // Rating buttons
    const rats = document.createElement("div");
    rats.className = "rats";
    ["t","l"].forEach(function(type){
      const rg = document.createElement("div");
      rg.className = "rg";
      rg.innerHTML = '<label>'+(type==="t"?"转写质量":"标签准确度")+'</label>';
      ["good","ok","bad"].forEach(function(val){
        const btn = document.createElement("button");
        btn.className = "btn"+(r[type]===val?" "+CLASSES[val]:"");
        btn.innerHTML = EMOJI[val]+" "+LABELS[val];
        var _id=id, _type=type, _val=val;
        btn.onclick = function(){setR(_id,_type,_val)};
        rg.appendChild(btn);
      });
      rats.appendChild(rg);
    });
    card.appendChild(rats);

    // Notes
    const inp = document.createElement("input");
    inp.className = "ni";
    inp.placeholder = "备注（背景噪声、口音、需要关注的错误）";
    inp.value = N[id] || "";
    inp.onchange = function(){setN(_id,this.value)};
    var _id=id;
    inp.onchange = function(){setN(_id,this.value)};
    card.appendChild(inp);

    c.appendChild(card);
  });
  document.getElementById("pf").style.width = Math.round(tc/50*100)+"%";
  document.getElementById("stats").innerHTML = '<div>✅ 转写: '+tc+'/50</div><div>✅ 标签: '+lc+'/50</div>';
}

function setR(id,type,val){
  if(!R[id]) R[id]={};
  R[id][type] = (R[id][type]===val) ? undefined : val;
  localStorage.setItem("r",JSON.stringify(R));
  render();
}
function setN(id,val){N[id]=val;localStorage.setItem("n",JSON.stringify(N));}
function saveProgress(){
  localStorage.setItem("r",JSON.stringify(R));
  localStorage.setItem("n",JSON.stringify(N));
  document.getElementById("sv").textContent = "✅ 已保存";
  setTimeout(function(){document.getElementById("sv").textContent = "";},3000);
}
function exportResults(){
  var out = CARDS.map(function(item){
    var r = R[item.id] || {};
    return {id:item.id, speaker:item.speaker, emotion:item.emotion, gt_text:item.gt_text, whisper_text:item.whisper_text, transcription_rating:r.t||"unrated", label_rating:r.l||"unrated", notes:N[item.id]||""};
  });
  var blob = new Blob([JSON.stringify(out,null,2)], {type:"application/json"});
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "eatd_review_results.json";
  a.click();
}
document.getElementById("btnSave").onclick = saveProgress;
document.getElementById("btnExport").onclick = exportResults;
render();
</script>
</body>
</html>"""

with open("review_page/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"OK: {len(html)} chars")
