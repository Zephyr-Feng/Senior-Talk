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
