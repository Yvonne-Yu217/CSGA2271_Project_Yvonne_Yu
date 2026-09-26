# 实验复盘与研究方向决策

日期：2026-09-26。依据：HPC 提交 `6d67ef4`、`pilot/results/final_metrics.json`、原始报告及实现代码。本次没有连接 HPC 重新计算预测，也没有提交 GPU 作业。

## 执行更新：有效性门槛已经触发停止条件

随后在 HPC 上完成了预定的 P0–P2，而不是继续调参。冻结 deletion checkpoint 的 prediction ensemble 在 deletion/matched/natural 上的 Δ-Acc@1 分别为 79.08%/73.17%/56.01%，对应 inverse cosine 为 84.50%/75.75%/58.16%；冻结迁移没有翻转原负结果。

100 条分层样本全部通过带框审阅图逐条检查：44 条通过 target change、control preservation 和 grammar 三项，53 条失败，3 条不确定。审阅者是 Codex 视觉检查，**不是人工标注者**，因此 human-confirmed count 仍为 0。尽管如此，低于一半的通过率已经足以否定“现有自动标签可直接支持 confirmatory claim”。342 个历史区域中另有 11 个在 CLIP center crop 中完全不可见、141 个仅部分可见、19 个触发 nearest-patch fallback。

决策门选择方向 A：课程项目保留为严格的负结果、强基线比较和任务有效性分析；在修复或更换标签任务前，不运行两个 learned variants，不启动 untouched confirmation、Visual Genome 或 CVPR 扩展。后文 P3/P4 现在是被门槛明确停止的条件分支，而不是待执行清单。

## 我的建议

**停止把现有轻量 MLP 的性能优势当成既定研究主线。先完成一次规模受限的有效性检查，再决定保留问题、改成分析项目，或换题。** 不应为了维护原 proposal，不断增加模块和数据集来寻找偶然正结果。

现有结果不支持该模型优于简单基线，却还不能证明整个问题没有价值。自然 caption 的标签、测试含义和部分评测实现有缺陷；先解决这些缺陷，才有依据判断任务是否值得继续。课程项目可以凭严谨的负结果与分析成立；CVPR 级贡献目前没有得到支持。

## 1. 已有结果的准确解释

| 实验 | learned rank+locality+control | inverse cosine | 可以得出的结论 |
|---|---:|---:|---|
| 删除干预 Acc@1 | 77.22% | 84.50% | learned 低 7.28 个百分点；配对 95% CI 为 [-11.70, -3.33] 个百分点 |
| 长度匹配替换 Acc@1 | 74.86% | 75.75% | 差值区间跨零，尚无优势证据 |
| 自然 caption Acc@1 | 50.17% | 58.16% | learned 更低，但多区域标签存在问题，不能直接作最终语义结论 |

上述 learned 数值是三个 seed 的**指标平均**，不是先平均预测再排序的集成模型。删除实验有 100 张测试图、337 对；自然 caption 有 97 张测试图、251 对。候选框只有 2–6 个，平均约 2.41 个，因此删除实验随机处理并列分数的基线约为 43.86%，不能拿 1/大量区域作为机会水平。

学习模型打乱图像后降到 43.26%，支持它在这个设置中使用了视觉特征；这不代表它比 inverse cosine 提取了更好的信息。inverse cosine 是合理的强基线，目前没有证据称它靠数据捷径取胜。

原报告的绝对定位中，n-gram matching 达到 67.17%，learned 为 62.33%，inverse cosine 为 60.50%。但该指标代码在并列时偏向固定排在第一个的目标框，必须先统计并列频率并重算，才能确认这些差异。

## 2. 必须纠正的解释和实现

1. `make_report.aggregate()` 平均各 seed 已算出的 image-level metrics；`faithfulness.py` 才平均原始预测。两类结果不能都叫 ensemble。
2. 每个 `--minus-field` 都重新训练 scorer。自然 caption 和 matched 实验是各自训练后的评测，不是 deletion-trained 模型的跨干预泛化。SigLIP 也是换编码器重新训练，不是直接权重迁移。
3. 自然 caption 只排除了目标实体 ID 和原短语，没有保证其他框对应的事实仍被覆盖。给所有非目标区域施加 invariance 或 covered=0 标签可能错误。应建立每个事实在两段文本中的 coverage 状态，多目标变化允许多个答案。
4. 长度匹配替换反复使用 `unspecified`，且 `man → person` 可能保留粗实体含义。这个控制没有证明语法匹配，也没有严格固定“什么信息被删掉”。应指定测试的是 identity、细类别还是属性。
5. locality 数值降低可能部分来自输出幅度整体收缩。需看 score 分布、目标变化和无关变化的联合曲线、排序效果及校准后的比较；不能只挑较小的 off-target drift。
6. 绝对定位使用 `argmax`，目标框总在索引 0；并列时会偏向目标。采用期望随机破平局或打乱候选顺序并固定种子，MRR 同样处理。
7. 空间 baseline 使用整图 center crop，部分目标框可能被裁掉；代码会给没有可见 patch 的框选择最近 patch。crop baseline 却直接看目标 crop。先比较可见比例，分层报告，再讨论方法差异。
8. 现有 recoverability 只是同一 CLIP 对目标短语在均值颜色遮挡前后的 similarity 差，不等价于恢复事实或独立模型的可用信息。需要同面积对照、多种遮挡和独立 reader。
9. 特征缓存 fingerprint 未包含编码器 revision、processor 配置和图像哈希。更换模型且复用路径时存在误用缓存风险；新实验前必须补齐。
10. 最终账本包含仍在 RUNNING 的 job；1.1267 GPU hours 只是报告时快照。账本代码也默认一张 GPU，未来须按实际 AllocTRES 计数，并去除父作业/step 重复计费。

自动 audit 的 pass 说明结构、框、划分、文件哈希通过检查，**不是人工语义标注正确率为 100%**。上述问题不意味着所有现有数字无效；删除干预的 seed-mean 比较仍是有价值的负结果。

## 3. 三个候选方向与选择条件

### A. 保留课程题目，改为严格的实验分析（当前最稳妥）

研究问题：常见 region similarity 和 attribution 方法，在什么条件下能识别文本遗漏的视觉事实？哪些看似漂亮的干预指标会误导判断？

贡献可由独立覆盖标签、强基线比较、自然 caption 多目标修正、尺度/裁剪/并列偏差分析组成。无需发明一个必胜的新模型。这是诚实、可执行的课程研究方向，但现有规模和发现尚不能据此声称 CVPR novelty。

### B. 保留问题，检验“基于语义覆盖的区域选择”（有条件继续）

先用 region caption + entailment 或 phrase matching 建立直接基线；将区域细化到 entity、attribute、relation 的可判定事实。只在它们出现稳定、有意义的失败时，考虑小型改进。

最多比较两种有明确动机的改法：相对最强基线学习一个有界 residual；或用覆盖标签训练 scorer，并用该任务的验证指标选 checkpoint。比较同等训练数据，做样本量曲线和必要 loss 消融。不能从现有负结果直接推断“必须换更大的模型”。

### C. 如果保留问题没有进展，转向“预算内补充哪些视觉事实”（备选换题）

给定已有 caption，每次只允许查看/描述一个 region，比较哪种选择策略带来最多**新增且正确**的事实。保持 region 数、像素/文本预算和 captioner 相同，比较 random、面积、saliency、inverse similarity、phrase/entailment、学习策略。

这有更明确的用途，但不是现有实验已经支持的新方向。开始前必须重新查相关工作，并用少量人工核验样本验证 region selection 确实影响新增信息。若简单策略已足够、改善不可测或读者评估不可靠，也应放弃，不能靠改标题把它包装为成功。

## 4. 一次有退出条件的继续实验

| 阶段 | 最小产出 | 决策 |
|---|---|---|
| P0：证据修复（优先 CPU） | 修正统计用语；恢复 raw predictions 和最终账本；评估 ties/crop visibility/cache 身份 | 无法获得关键原始产物时先明确缺失，不能虚构重算 |
| P1：100 对分层语义审核 | 每区域/事实 coverage、contradiction、unknown；自然对的多目标集合；标注政策 | 歧义或误标太多就收窄任务，停止扩大模型 |
| P2：锁定模型测试 | deletion 训练一次，权重不变评估 matched/natural；与各自重训分开；同图配对 CI | 判断困难来自标签、干预分布还是模型本身 |
| P3：有限模型比较 | strongest baseline、当前 MLP、至多两种针对性改法；先 validation；记录 learning curves | 没有稳定改善则停止方法扩张，走 A 或提议 C |
| P4：确认与剩余 scope | 新的未用测试图；必要的 VG、自动区域、独立 reader 检查；完成原 proposal 的证据矩阵 | 哪项未做就注明；不把 encoder 重训当整个 RQ3 已完成 |

P0/P1 可立即做准备；P2/P3 先预估不超过 **4 个新增 GPU hours 的内部检查上限**，这是阶段总预算内的诊断投入，不是要求用户另开短时节点或追加额度，也不保证届时全部完成。若最终账本显示余额不足，则缩小/停止。禁止把全部剩余额度预先分配给无限超参搜索。

旧测试集已参与方向选择，今后视为 exploratory。新的确认集需与此前全部样本按图像隔离。预先确定 primary metric、实际有意义的效应和停止条件，不能用哪个测试指标有利就临时改主张。

继续学习方法的条件：在可信覆盖标签上，paired CI 支持相对最强直接基线的改进，且在锁定模型的文本改写上保持一致。否则课程采用 A；如果用户希望继续追求方法论文，则报告失败并建议 C 或重新选题，不硬把当前模型推向 CVPR。

## 5. GPU 利用率与连续推进

历史日志只有 formal/siglip/faithfulness 的 19/7/20 次采样，均值为 16.2%/15.9%/3.9%。这说明观测窗口占用偏低，但不能据此断定全程浪费，也不能诊断唯一根因。代码中的同步 PIL 解码、重复模型加载/特征提取、频繁 CPU 同步、占着 GPU 做 bootstrap 都是待测瓶颈。

使用当前可用 GPU，不绑定品牌或 partition。每 5 秒记录 GPU utilization、VRAM、CPU、吞吐、阶段和 job ID；分别报告初始化、GPU 推理、训练、CPU 后处理及总作业时间。稳定计算阶段尽量达到约 70–90% 的有效占用；若连续 2 分钟低于 30% 或吞吐停止，马上定位输入/同步/小任务瓶颈。这是内部诊断标准，不是对集群自动回收政策的断言。

提前准备下一批有必要的任务。通过 batching、workers、prefetch、缓存复用和合理的 CPU/GPU 重叠保持推进。可让 subagent 并行审核数据、准备独立 baseline 或分析上一轮结果；主 agent 统一调度 GPU 和记账。只有测得有内存余量且提高有效吞吐时才同卡并发。没有有效 GPU 任务时保存状态并释放节点，不制造占用、不重复已完成计算。

## 6. 文件与范围

原始结果保存在 `pilot/results/final_metrics.json`；历史报告保留数值并修正解释。更新后的 proposal 会明确试验性假设、既有负结果及换方向条件。CVPR extension 文档仅作为后续候选，不是本轮承诺。

仓库 HPC 记录的上限为 24 GPU hours，早先本地计划是 20；下一次运行应从 HPC handoff 核实授权与最终消耗，未解决时按较低的 20 小时总额执行，不能把报告快照当余额。本次完成证据审查、计划、文档和下一轮可运行代码，不扩充额度、不提交 GPU 作业。运行入口为 `pilot/next_round.py`，分 CPU preflight / GPU inference / CPU summarize；具体命令见 `hpc/NEXT_ROUND.md`。P3/P4 为有条件的后续方向，尚未作为自动运行任务实现。
