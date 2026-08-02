#!/usr/bin/env python3
"""Generate the review HTML page with embedded data."""
import json

with open("/d/Claude Code/Code/tmp_review_results.json") as f:
    results = json.load(f)

data_json = json.dumps(results, ensure_ascii=False)

html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>语音模块 人工核对 — 50段音频评审</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #f5f5f5; padding: 20px; }
h1 { margin-bottom: 10px; }
.progress { margin: 15px 0; }
.progress-bar { height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; }
.progress-fill { height: 100%; background: #4CAF50; width: 0%; transition: width 0.3s; }
.stats { display: flex; gap: 20px; margin: 10px 0 20px; flex-wrap: wrap; }
.stats div { background: #fff; padding: 8px 16px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.card { background: #fff; border-radius: 8px; margin-bottom: 16px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.card-id { font-weight: bold; font-size: 16px; }
.card-meta { color: #666; font-size: 13px; }
.audio-player { margin: 10px 0; }
.audio-player audio { width: 100%; max-width: 400px; }
.text-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0; }
@media (max-width: 700px) { .text-compare { grid-template-columns: 1fr; } }
.text-box { padding: 10px; border-radius: 6px; font-size: 14px; line-height: 1.6; }
.gt-box { background: #e8f5e9; border-left: 3px solid #4CAF50; }
.whisper-box { background: #fff3e0; border-left: 3px solid #FF9800; }
.text-label { font-weight: bold; font-size: 12px; text-transform: uppercase; margin-bottom: 4px; }
.evidence { margin: 8px 0; }
.evidence-tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
.evidence-tag.sleep_complaint { background: #e3f2fd; color: #1565c0; }
.evidence-tag.loneliness { background: #fce4ec; color: #c62828; }
.evidence-tag.anxiety_worry { background: #fff3e0; color: #e65100; }
.evidence-tag.loss_of_interest { background: #f3e5f5; color: #6a1b9a; }
.evidence-tag.repeated_questions { background: #e8eaf6; color: #283593; }
.evidence-tag.time_confusion { background: #efebe9; color: #4e342e; }
.ratings { display: flex; gap: 20px; margin: 10px 0; flex-wrap: wrap; }
.rating-group { display: flex; align-items: center; gap: 8px; }
.rating-group label { font-size: 13px; font-weight: bold; min-width: 80px; }
.btn-group { display: flex; gap: 4px; }
.btn { padding: 4px 12px; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; font-size: 13px; background: #fafafa; }
.btn:hover { background: #f0f0f0; }
.btn.selected-good { background: #4CAF50; color: white; border-color: #4CAF50; }
.btn.selected-ok { background: #FFC107; color: white; border-color: #FFC107; }
.btn.selected-bad { background: #f44336; color: white; border-color: #f44336; }
.notes-input { width: 100%; margin-top: 8px; padding: 6px 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; }
.actions { position: sticky; bottom: 0; background: #fff; padding: 12px 20px; box-shadow: 0 -2px 10px rgba(0,0,0,0.1); display: flex; gap: 12px; align-items: center; border-radius: 8px 8px 0 0; }
.actions button { padding: 8px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }
.btn-save { background: #2196F3; color: white; }
.btn-export { background: #4CAF50; color: white; }
.btn-reset { background: #f44336; color: white; }
.safety-flag { color: #f44336; font-weight: bold; }
</style>
</head>
<body>
<h1>🎤 语音模块转写与语义标签 人工核对</h1>
<p>共50段音频 · 听音频 → 对比转写和标签 → 打分</p>
<p style="color:#666;font-size:13px;">GT = 数据集原始标注（参考标准） | Whisper = 系统转写结果</p>

<div class="stats" id="stats"></div>
<div class="progress">
  <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
</div>
<div id="cards"></div>

<div class="actions">
  <button class="btn-save" onclick="saveProgress()">💾 保存进度</button>
  <button class="btn-export" onclick="exportResults()">📤 导出结果</button>
  <button class="btn-reset" onclick="resetAll()">🔄 重置</button>
  <span id="saveStatus" style="color:#666;font-size:13px;"></span>
</div>

<script>
const DATA = __DATA_JSON__;

let ratings = JSON.parse(localStorage.getItem('eatd_review_ratings')) || {};
let notes = JSON.parse(localStorage.getItem('eatd_review_notes')) || {};

function render() {
    const container = document.getElementById('cards');
    container.innerHTML = '';
    let completed = 0;
    DATA.forEach((item) => {
        const id = item.id;
        const rated = ratings[id] && (ratings[id].transcription || ratings[id].label);
        if (rated) completed++;

        const evidence = item.semantic_evidence || [];
        const safetyFlag = item.safety_flag;

        let evidenceHtml = '';
        if (evidence.length > 0) {
            evidenceHtml = '<div class="evidence">' + evidence.map(e =>
                '<span class="evidence-tag ' + e.label + '">' + e.label_zh + '</span>'
            ).join('') + '</div>';
        }
        if (safetyFlag) {
            evidenceHtml += '<span class="safety-flag">&#9888;&#65039; 敏感词触发</span>';
        }

        const idStr = String(id).padStart(2, '0');

        cardHtml = '<div class="card" id="card-' + id + '">'
            + '<div class="card-header">'
            + '<div><span class="card-id">#' + id + '</span> <span class="card-meta">' + item.speaker + ' / ' + item.emotion + ' / ' + item.duration + 's / SDS=' + (item.sds ?? 'N/A') + '</span></div>'
            + '<div><span class="card-meta">语速:' + (item.speech_rate ? item.speech_rate.toFixed(1) : '?') + '字/秒 停顿:' + (item.pause_ratio ? Math.round(item.pause_ratio * 100) : '?') + '%</span></div>'
            + '</div>'
            + '<div class="audio-player"><audio controls src="audio/' + idStr + '_' + item.speaker + '_' + item.emotion + '.wav"></audio></div>'
            + '<div class="text-compare">'
            + '<div class="text-box gt-box"><div class="text-label">&#128215; GT 原文</div>' + escHtml(item.gt_text) + '</div>'
            + '<div class="text-box whisper-box"><div class="text-label">&#128217; Whisper 转写</div>' + (item.whisper_text ? escHtml(item.whisper_text) : '<i style="color:#999">(空)</i>') + '</div>'
            + '</div>'
            + evidenceHtml
            + '<div class="ratings">'
            + '<div class="rating-group"><label>转写质量</label>'
            + '<div class="btn-group">'
            + btnHtml(id, 'transcription', 'good', '&#9989; 准确')
            + btnHtml(id, 'transcription', 'ok', '&#9888;&#65039; 部分错')
            + btnHtml(id, 'transcription', 'bad', '&#10060; 基本不可用')
            + '</div></div>'
            + '<div class="rating-group"><label>标签准确度</label>'
            + '<div class="btn-group">'
            + btnHtml(id, 'label', 'good', '&#9989; 正确')
            + btnHtml(id, 'label', 'ok', '&#9888;&#65039; 部分对')
            + btnHtml(id, 'label', 'bad', '&#10060; 错误/漏标')
            + '</div></div>'
            + '</div>'
            + '<input class="notes-input" placeholder="备注（背景噪声、口音、需要关注的错误）" value="' + escHtml(notes[id] || '') + '" onchange="setNote(' + id + ', this.value)">'
            + '</div>';

        container.innerHTML += cardHtml;
    });

    // Re-attach event handlers via onclick attributes already set
    updateProgress();
}

function btnHtml(id, type, value, label) {
    const r = ratings[id] ? ratings[id][type] : null;
    const sel = r === value ? ' selected-' + value : '';
    return '<button class="btn' + sel + '" onclick="setRating(' + id + ",'" + type + "','" + value + "')\">" + label + '</button>';
}

function escHtml(s) {
    if (!s) return '';
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function setRating(id, type, value) {
    if (!ratings[id]) ratings[id] = {};
    ratings[id][type] = (ratings[id][type] === value) ? null : value;
    localStorage.setItem('eatd_review_ratings', JSON.stringify(ratings));
    render();
}

function setNote(id, value) {
    notes[id] = value;
    localStorage.setItem('eatd_review_notes', JSON.stringify(notes));
}

function updateProgress() {
    const fullDone = Object.values(ratings).filter(r => r.transcription && r.label).length;
    const pct = Math.round(fullDone / 50 * 100);
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('stats').innerHTML =
        '<div>&#9989; 已评转写: ' + Object.values(ratings).filter(r=>r.transcription).length + '/50</div>'
        + '<div>&#9989; 已评标签: ' + Object.values(ratings).filter(r=>r.label).length + '/50</div>'
        + '<div>&#128221; 全部完成: ' + fullDone + '/50</div>';
}

function saveProgress() {
    localStorage.setItem('eatd_review_ratings', JSON.stringify(ratings));
    localStorage.setItem('eatd_review_notes', JSON.stringify(notes));
    document.getElementById('saveStatus').textContent = '&#9989; 已保存于浏览器本地存储';
    setTimeout(() => document.getElementById('saveStatus').textContent = '', 3000);
}

function exportResults() {
    const output = DATA.map(item => {
        const r = ratings[item.id] || {};
        return {
            id: item.id,
            speaker: item.speaker,
            emotion: item.emotion,
            duration: item.duration,
            gt_text: item.gt_text,
            whisper_text: item.whisper_text,
            evidence_count: item.evidence_count,
            transcription_rating: r.transcription || 'unrated',
            label_rating: r.label || 'unrated',
            notes: notes[item.id] || ''
        };
    });
    const blob = new Blob([JSON.stringify(output, null, 2)], {type: 'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'eatd_review_results.json';
    a.click();
}

function resetAll() {
    if (confirm('确定重置所有评分？')) {
        localStorage.removeItem('eatd_review_ratings');
        localStorage.removeItem('eatd_review_notes');
        ratings = {}; notes = {};
        render();
    }
}

render();
</script>
</body>
</html>"""

# Replace placeholder with actual JSON data
html = html.replace("__DATA_JSON__", data_json)

with open("/d/Claude Code/Code/tmp_review_page.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML generated: {len(html)} chars ({len(results)} items)")
