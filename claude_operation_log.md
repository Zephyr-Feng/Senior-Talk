# Claude Code 操作日志

### [2026-08-04] - 抑郁报告中增加孤独倾向提示（阶段一：本地代码）

* **当前操作动作**：创建新模块 / 修改训练数据报告生成器 / 运行单元测试
* **核心变更说明**：
  1. 新增 `scripts/loneliness_signal.py`：ULS-8 六维度关键词映射（缺乏陪伴/无人倾诉/被冷落/无法融入/孤单感受/被遗弃）+ 否定/适应语境过滤 + 维度计数强信号判定（≥2 维度或 ≥1 维度含明确词汇），严格阈值避免误报
  2. 修改 `scripts/prepare_training_data.py` 的 `generate_reasoning_output()`：孤独信号明显时，文本方面追加"提示可能存在孤独情绪"、建议追加"评估 ULS-8/鼓励社交互动"；**结论行保持抑郁主判断不变**（`结论：需要关注/正常`），保证 eval_lora.py 解析逻辑零改动
  3. `build_conversation()` 新增 `lonely_hint` 字段；数据准备输出全量/训练集孤独信号统计
* **涉及/修改的文件清单**：
  - `scripts/loneliness_signal.py` (Created)
  - `scripts/prepare_training_data.py` (Modified)
  - `tmp_test_loneliness.py` (Created)
* **执行结果与验证状态**：`python -m py_compile` 三个文件全绿；`tmp_test_loneliness.py` 12/12 用例通过（含 t_68「一个人生的一块白板」误报案例 → 不触发，多维度/丧偶场景 → 触发）
* **置信度或遗留待办（TODO）**：待同步服务器跑数据生成（无卡 CPU）；重训 LoRA 前需用户 GPU 授权；孤独提示效果分析待微调后统计
---
### [2026-08-04] - 孤独规则引擎全量验证 + 融合消融实验启动

* **当前操作动作**：运行全量语料验证 / 编写融合消融实验脚本 / 启动全量 LoRA 推理（GPU 已授权）
* **核心变更说明**：
  1. **孤独规则引擎 EATD 全量验证**：479 条转写文本强信号 **0 条**（EATD 为任务式朗读语料，不含孤独内容）；18 条"缺乏陪伴"维度命中中 **17 条误报**（"一个人"作不定代词/积极语境的多义性）、**1 条真信号**（t_72 "一个人过年不开心"）因单维度被阈值漏掉 → 强信号阈值成功阻止误报升级，保守设计生效；规则召回无法在 EATD 上验证，需中老年自然对话语料
  2. **声学通道分类器**：486 条按人 GroupKFold out-of-fold 概率，RandomForest **AUC=0.4756** / SVM 0.4152 / GB 0.4736 → EATD 上声学单独无区分力（正样本 159/486，Sens 8.8%），融合中文本通道为主力
  3. 新增 `tmp_eval_lora_full.py`：LoRA 报告版（lora_adapter_dep_v1）全量 486 条推理，4bit 量化 + 断点续跑 + 逐条写盘，为融合实验准备 P_text（结论二值）
  4. 新增 `tmp_fusion_acoustic.py` / `tmp_fusion_ablation.py`：决策级融合 P_fused(w)=w·P_acoustic+(1-w)·P_text，w∈[0,1] 网格搜索 + 消融表（纯声学/纯LoRA/最优融合）+ 零样本 63 条参照
* **涉及/修改的文件清单**：
  - `scripts/tmp_diag_loneliness_hits.py` (Created)
  - `scripts/tmp_eval_lora_full.py` (Created)
  - `scripts/tmp_fusion_acoustic.py` (Created)
  - `scripts/tmp_fusion_ablation.py` (Created)
  - 服务器：`training_output/fusion/p_acoustic.json` (Created)
* **执行结果与验证状态**：tmp_eval_lora_full 5 条小批量验证通过（模型加载/推理/断点续跑正常，80% 准确）；全量 474 条后台推理进行中（~42s/条 ≈ 5.5h）；tmp_fusion_acoustic 已完成输出
* **置信度或遗留待办（TODO）**：eval_full 完成后跑 tmp_fusion_ablation.py 出消融表；零样本 63 条与 486 条评估集不一致，仅作参照非同集对比；LoRA 训练数据含 EATD 文本改造（合成 400 条），测试集存在部分文本泄漏风险，报告中需注明
---
### [2026-08-05 00:40] - 融合消融实验完成 + 服务器自动关机

* **当前操作动作**：运行全量融合消融实验 / 自动关机
* **核心变更说明**：
  1. LoRA 报告版全量 479 条推理完成（双分片并行，断点续跑，00:39 全部完成）
  2. 融合消融结果（479 条，正 154/负 325）：
     - 纯声学分类器 RandomForest：Acc 63.7% / Sens 9.1% / F1 13.9% / **AUC 0.471**（接近随机）
     - 纯 LoRA 文本：Acc 73.9% / Sens 70.8% / Spec 75.4% / F1 63.6% / **AUC 0.731**
     - 决策级融合最优 w=0.0（=纯 LoRA）：声学无任何增量 → 融合退化为纯文本
     - 参照零样本 63 条：Sens 25% → LoRA 微调后 70.8%（**+45.8pp**）
  3. 结论：EATD 上声学特征无信息量（任务式朗读语料声学差异小 + 8 个全局统计特征过粗）；LoRA 文本通道是绝对主力；微调价值巨大
* **涉及/修改的文件清单**：
  - 服务器：`training_output/fusion/ablation_result.json` (Created)
  - 服务器：`training_output/eval_full/eval_full*.jsonl` (Created, 479 条)
* **执行结果与验证状态**：融合实验完成 ✅；结果文件 16374B 确认存在后执行 `shutdown now` ✅；3 秒后重连 Connection refused 验证关机成功 ✅
* **置信度或遗留待办（TODO）**：声学无增量的根因待分析（时序特征/eGeMAPS/DAIC-WOZ 可能改善）；融合未提升的结论需在报告中如实呈现
---
### [2026-08-07] - 孤独倾向 LoRA 训练链路搭建（本地，未开卡）

* **当前操作动作**：合成孤独数据生成器 + 训练/评估脚本（纯本地工作，零 GPU 成本）
* **核心变更说明**：
  1. `gen_lonely_samples.py`：模板注入法合成孤独样本（ULS-8 六维度覆盖），标签随生成已知（无需 AI 标注）
  2. 全量生成 300 正 + 300 负 = 600 条，质检通过：负样本误报 0 条（硬性检查，>0 报错退出）、文本唯一率 100%、六维度分布均衡（88~113）、长度 11-72 字
  3. 抽查未通过规则识别的 105 条正样本：全部为高质量含蓄表达（"想说点心里话，翻翻手机也不知道发给谁"等），标签正确——恰是规则引擎盲区，训练 MLLM 语义判断的价值点
  4. `train_lora_lonely.py`：QLoRA 4bit + r16/alpha32（与抑郁版同配置），600→540 条训练（分层留出 30+30 验证集 → holdout.jsonl）
  5. `eval_lonely_full.py`：双评估集（heldout 60 条主评估 + EATD 479 条误报检查），断点续跑 + 分片支持
* **涉及/修改的文件清单**：
  - `scripts/gen_lonely_samples.py` (Created)
  - `scripts/train_lora_lonely.py` (Created)
  - `scripts/eval_lonely_full.py` (Created)
  - `training_data_lonely/lonely_samples.jsonl` (Created, 600 条) + `quality_report.txt`
  - `training_data_lonely/` 为 git 已提交（commit 50ce2c1）
* **执行结果与验证状态**：语法检查通过 ✅；本地 300+300 生成 + 质检通过 ✅；尚未开卡（等待用户授权）
* **置信度或遗留待办（TODO）**：开卡批次 = 上传数据+脚本 → `train_lora_lonely.py`（~5h）→ `eval_lonely_full.py heldout`（~30min）+ `eatd`（双分片 ~2h）→ 自动关机；孤独版训练/评估均无金标准，评估以 heldout + 误报率 + 人工抽查为准
---
### [2026-08-07] - 孤独 LoRA 开卡批次执行完成 + 解析 bug 修复

* **当前操作动作**：开卡批次全流程执行（训练→评估→自动关机）+ 统计解析 bug 修复
* **核心变更说明**：
  1. 批次 12:01 启动：训练 340 步 × 7.4s ≈ 42min（12:43 完成，远快于预估 5h）→ heldout 评估 ~3min → EATD 双分片 ~10min → 12:53 汇总 → **自动关机**（13:03 连接确认）
  2. 初判结果异常（EATD FP=478）→ 根因定位：**`"明显" in "不明显"` 子串匹配 bug**（Python 子串包含误判）→ 新增 `parse_lonely_pred()` 精确匹配（"明显"/"不明显"）
  3. **修复后真实结果**：
     - heldout（30+30）：TP=29 TN=30 FP=0 FN=1 → **Acc 98.3% / Sens 96.7% / Spec 100% / F1 98.3%**
     - EATD 误报检查（479 条真实语料）：383/479 判"不明显" ✅，**误报 96/479（20%）**
     - FN=1 为生成格式异常（输出截断 + `| assistant |` 重复，全批次共 3 条此类），非语义漏检
  4. EATD 误报抽查分析：多数为抑郁语料负面人际表达（争吵/讨厌的人）被泛化为孤独信号——部分合理（人际疏离），少数幻觉（"不喜欢小孩子"→"被遗弃"）
* **涉及/修改的文件清单**：
  - `scripts/eval_lonely_full.py` (Modified)：新增 parse_lonely_pred 精确匹配 + 汇总从结论原文重解析
  - `scripts/summarize_lonely.py` (Modified)：同上，兼容旧 jsonl 重算
  - 服务器：`training_output/eval_lonely/{summary_eatd,summary_heldout}.json` (Recomputed)
* **执行结果与验证状态**：本地重解析与服务器重跑结果一致 ✅（heldout 98.3% / EATD FP 96）；关机成功 ✅（13:03 连接拒绝，用户开卡后重连确认结果文件完整）
* **置信度或遗留待办（TODO）**：EATD 20% 误报的阈值调优（负面情绪 vs 孤独的边界）；生成格式异常（3 条）根因待查（max_new_tokens/解码参数）；孤独模型评估无金标准，以 heldout + 误报率 + 人工抽查为准
