cat >> /root/autodl-tmp/senior_project/CLAUDE.md << 'EOF'

## 🚀 服务器使用规则

### 无卡启动（默认）
- **默认状态**：无卡启动（无 GPU），只跑 CPU 任务（tiny/small Whisper、sklearn 训练、特征提取）
- **开卡条件**：仅在明确告知用户"需要 GPU"并得到确认后才开卡。不要擅自让用户开卡。
- **适用场景**：特征提取、轻量分类器训练、长音频测试等 CPU 可完成的任务
- 开卡后需更新本文件中的配置（`DEVICE`、`WHISPER_MODEL_SIZE`、`compute_type`）

### EATD-Corpus 数据集
- 来源：ICASSP 2022 抑郁症检测挑战赛（中文语音+文本+抑郁SDS标签）
- 下载：OneDrive 密码分享（密码 Ymj26Uv5）
- 数据位置：/root/autodl-tmp/senior_project/data/EATD-Corpus/
- 结构：t_*（训练集83人）, v_*（验证集79人），每人3段情绪音频+文本+SDS评分
- 用途：验证 B2 声学特征区分度 + 训练抑郁分类器
EOF
echo "Done"