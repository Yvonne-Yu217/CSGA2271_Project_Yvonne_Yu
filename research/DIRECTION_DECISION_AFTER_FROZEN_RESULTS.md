# 最新结果后的方向判断

依据：仓库 `5f24bd4`，2026-09-26 拉取；本次额外核对一手论文、官方仓库和实现入口。没有提交 HPC 作业。

## 结论

**停止把“CLIP crop 特征 + 小 MLP + 删词差分损失”当作继续扩大的主线。保留图片相对已有文字还能补充什么信息这个问题，建议更换任务目标和监督方式，再做有明确成功条件的验证。**

用户现在要求课程项目也有正向结果与新意，因此“整理负结果就算完成项目”不再满足目标。旧实验作为前期探索保留。更多 HPC 资源能扩大有效实验，但无法自动弥补目标不匹配或与已有工作的重复。

推荐的新候选：**根据已经写出的描述，学习下一次该看哪个区域，才能补充最多有视觉证据的新事实；重点包括已经提到的物体仍然遗漏的属性和关系。** 完整方案见 [后续 proposal](FOLLOWUP_PROPOSAL.md)。这是值得验证的研究假设，尚无新实验结果，也不保证顶会录用。

## 1. 最新结果究竟说明什么

下表是固定 deletion-trained checkpoint 后真正的迁移实验。Learned 使用三个 seed 的预测 ensemble；最后一列是相对 inverse cosine 的配对差值，单位为百分点。

| 测试 | Inverse cosine Δ-Acc@1 | Learned ensemble | 差值及 95% CI |
|---|---:|---:|---|
| 删除 | 84.50% | 79.08% | −5.42 [−10.42, −0.67] |
| 长度匹配 | 75.75% | 73.17% | −2.58 [−7.92, +2.75] |
| 自然 caption | 58.16% | 56.01% | −2.15 [−4.98, +0.60] |

没有出现可信的 learned 优势。只看单条 caption 的绝对区域选择，删除集 learned ensemble 62.33%，inverse 60.50%，差值 CI [−3.50, +7.25]；也不能当成已证明的正向结果。Matched / natural 的绝对选择仍较弱。

筛出的有效小子集只有 10 / 25 / 9 对、9 / 22 / 9 张图。部分差分指标转正，但区间跨零，不能据此拯救主张。现有训练只用了 200 张图，所以这些结果否定的是当前规模下的实现与主张，不是证明所有学习方法都不可能成功。

### 必须修正上一份报告的语气

“44/100 通过全部审核”不等于“56% 数据的语义标签错误”。这是一份 Codex 视觉诊断，按干预与候选框数分层抽样，不是随机人口比例，更不是独立人工 gold。

| 干预 | 审核数 | 全部通过 | 目标变化 yes/no/uncertain | 控制区域失败 | 语法失败 |
|---|---:|---:|---|---:|---:|
| 删除 | 33 | 10 | 33 / 0 / 0 | 0 | 23 |
| 匹配 | 34 | 25 | 30 / 2 / 2 | 0 | 8 |
| 自然 | 33 | 9 | 25 / 7 / 1 | 23 | 0 |

删除组的 23 个失败按记录全是语法问题。自然组确实有多个事实同时改变的问题。审核还存在不一致：部分残句被判为语法通过，同类不定冠词错误有时通过有时失败。下一阶段需要重新裁定，不能把这些自动审核结果当成可靠标签真值。

## 2. 旧方法为什么可能从根上学错了目标

我们希望模型在只看到一条 caption 时，选出最值得补充的区域。当前损失主要教它：同一个区域在删词前后的分数应该有差别。

这两件事不等价。某区域的分数从 0.9 降到 0.8，另一个从 0.4 降到 0.1；第二个变化更大，但第一条 caption 下模型仍会选第一个。给每个区域增加一个与 caption 无关的偏置，差分可以不变，绝对排序却会变。Sigmoid 范围约束不消除这个目标不匹配。

此外，inverse cosine 的差分本身就是 `0.5 × v · (t_plus − t_minus)`。在人工删词任务中，文本差向量直接提供了被删内容的线索。因此这个基线很强有合理数学原因；不能把它叫“靠捷径作弊”。

**应该直接监督：选这个区域后，确实增加了多少正确且此前未表达的信息。** 同时检查模型是不是学会了跨 caption 的区域优先级，而非记住区域显著性或删词模板。

## 3. 文献对新意的限制

下列工作直接影响方向判断。这里只概括各自与我们相关的部分；论文宣称的性能没有在本地重现。

| 工作 | 已覆盖的内容 | 对我们的约束 |
|---|---|---|
| [CompreCap, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Lu_Benchmarking_Large_Vision-Language_Models_via_Directed_Scene_Graph_for_Comprehensive_CVPR_2025_paper.pdf) | 对象、属性、关系的详细 caption 评估 | “定义语义覆盖率”不是新贡献 |
| [CapRL, ICLR 2026，官方实现](https://github.com/InternLM/CapRL) | 用文本 reader 的 QA 表现训练 caption；仓库还有 CapRL++ | “让 caption 帮 reader 回答更多问题”不是新贡献 |
| [CCCaption, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_CCCaption_Dual-Reward_Reinforcement_Learning_for_Complete_and_Correct_Image_Captioning_CVPR_2026_paper.html) | 联合奖励完整性与正确性 | “补漏 + 降幻觉 + RL”已经有人做 |
| [ClaimDiff-RL](https://arxiv.org/abs/2605.20278) | 用图像验证 caption 之间的原子事实差异 | 原子事实差分本身也不足以作为 novelty |
| [AdaptVision, CVPR 2026](https://arxiv.org/abs/2512.03794) | 学习按需要获取高分辨率 crop，权衡答案与计算 | “自适应裁剪 + 成本/停止”不是新贡献 |
| [Generating Accurate and Detailed Captions for High-Resolution Images](https://arxiv.org/html/2510.27164v1) | 从初始 caption 寻找遗漏物体，检测、放大、生成区域描述再合并 | 原先泛泛的“找遗漏区域再补 caption”备选方向也已被覆盖 |
| [SC-Captioner, ICCV 2025，官方实现](https://github.com/zl2048/SC-Captioner) | caption 自我纠正与训练 | 不能只靠反思/改写循环声称方法创新 |
| [CapProbe](https://arxiv.org/abs/2608.11074) | 区域对齐的密集事实问答评估 | 区域 QA 覆盖不是新任务定义；发布状态需核实 |

有空间继续验证的更具体假设是：**相同图片在不同已有描述下，需要不同的后续视觉证据；直接学习这种上下文相关的动作收益，能比物体遗漏规则、语义匹配及通用 VLM 规划器更可靠地补充属性/关系，并迁移到另一种 reader。** 这是本次检索后的候选区别，不是穷尽检索证明的“首次”。

## 4. 是否值得继续：分开回答

- **旧 MLP 路线：不建议继续作为主线。** 一次规范监督/学习曲线对照可作为诊断，但不能让项目继续依赖其翻盘。
- **课程项目：建议尝试新任务的正向验证。** 采用强视觉模型、真实动作收益监督和清晰的对照，目标更直接，也能复用原来的区域与统计工具。必须超过有竞争力的 baseline；只超过 random 不满足用户要求。
- **CVPR / TPAMI 研究：目前不 ready，有条件值得探索。** 需要证明新方法相对相近工作有独立贡献，且增益来自所提出机制。增加 backbone、数据集数量或论文篇幅本身不够。
- **若 oracle 都不能比强 baseline 多补有价值的事实，或者强 prompting 已吃掉几乎全部收益：换题。** 不建议进入大规模 RL 来寻找偶然改善。

正向结果不能事先保证。我们可以通过“先证明确实有可利用的提升空间，再训练直接优化该目标的模型”提高成功概率，并在证据不支持时尽早换方向。

## 5. 已查看的仓库与复现状态

| 官方仓库 | 本次确认 | 使用方式与限制 |
|---|---|---|
| [CompreCap](https://github.com/LuFan31/CompreCap) | 读了 `evaluate.py`：对象匹配、属性/关系 evaluator、mask coverage 路径确实存在 | 外部评测起点；实例区分与 judge 偏差仍需审计 |
| [CapRL](https://github.com/InternLM/CapRL) | 读了 README 和 `Prism_Evaluation/Eval_CapRL.py`；有训练、QA 构建和模型链接 | 强 caption baseline 与 reader 评估参考；不能假定所有新旧训练 recipe 兼容 |
| [AdaptVision](https://github.com/AdaptVision/AdaptVision) | 读了实际训练脚本，包含 tool/area 奖励与 Qwen2.5-VL-7B 设置 | 参考动作接口与强 planner；现成训练配置需修改，含远程答案检查依赖 |
| [CCCaption](https://github.com/Shopee-MUG/CCCaption) | 公开仓库本次仅见 README | 有论文不等于有可直接复现的代码；若仍未发布，只能清楚标注自行适配 |
| [ClaimDiff-RL](https://github.com/ltl3A87/ClaimDiff-RL) | 有 reward-server/训练 recipe；benchmark suite 仍列为待发布 | 官方 recipe 依赖 Gemini；HPC 不能替代 API 和标注依赖 |
| [SC-Captioner](https://github.com/zl2048/SC-Captioner) | 有 LoRA SFT、自我纠正、评估命令及数据链接 | 可做改写 baseline；部分关系评估只提供问题 |

没有运行这些第三方训练脚本，没有安装进现有 pilot 环境。可运行文件存在只说明有复现起点，不代表复现已经完成。
