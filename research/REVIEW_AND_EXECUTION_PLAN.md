# Proposal review and execution plan

> Historical pre-experiment review. The current evidence-based plan is [RESULTS_REVIEW_AND_NEXT_STEPS.md](RESULTS_REVIEW_AND_NEXT_STEPS.md), with execution in [hpc/NEXT_ROUND.md](../hpc/NEXT_ROUND.md). Earlier scheduling or hardware assumptions below are superseded by GOAL.md.

> 2026-09-26 更新：用户已将第一目标改为在 HPC 上完整验证原 proposal。本文的课程收缩建议与 local pilot 只是历史评审建议；当前执行范围以 `../GOAL.md` 和 `../hpc/RESOURCE_PLAN.md` 为准，不能用 pilot 代替全局验证。

评审日期：2026-09-26。原稿：`proposal/What_Does_the_Image_Add_Proposal.tex`。本文是研究判断，不是课程教师的实际评分或会议录用预测。原稿已有未提交修改，评审与扩展另存，不覆盖原文件。

## 1. 总体评价

作为课程 research project，我给当前设计 **8/10 左右（有 A 档潜力，取决于执行）**：问题直观、可解释性动机清楚、冻结编码器适合有限算力、能通过负结果产生有价值的分析；把金融降为 optional 是正确的范围控制。当前主要问题是工作量偏大、语义定义不足、指标与训练目标过于同构，以及缺少最强的任务直接基线。

作为 CVPR 投稿构想，我认为**值得验证，但当前论证尚不足以支撑投稿**。如果交来的是只有这套 MLP、phrase removal 和 TIG 实验的完整论文，我倾向 weak reject：容易被解释为“识别 caption 没提到的 object”，方法简单，基准标签可能把任务答案编码进了编辑操作。CVPR 是会议；如以期刊为目标，可另考虑 TPAMI/IJCV，不能只把实验数量增多当作达到标准。

Novelty 判断分层：

| 层次 | 当前判断 | 需要补的证据 |
|---|---|---|
| 图文互补信息 | 已有明确先例，不能主张首次 | CIEA 等相关工作 |
| 区域级遗漏语义/覆盖率 | 有空间，但邻近工作显著 | CompreCap、CapProbe、dense captioning/grounding 对照 |
| 局部文本干预验证 | 有价值的组合，但一般 consistency loss 本身不强 | 同义改写不变、事实补全定向变化、无关变化稳定、视觉依赖 |
| 所提 MLP | 当前方法创新较弱 | 简单基线失败的具体机制，以及对应解决方案 |
| 严格 benchmark + 新发现 | 最可行的课程贡献，也可能发展为论文 | 独立标注、强基线、跨域重复结论、公开协议 |

## 2. 原 proposal 必须先完善的地方

### P0：重新定义预测对象

一个 region 同时有实体、颜色、动作、关系。`a dog` 已出现，并不意味着狗的颜色和动作已被覆盖；`an animal` 可能覆盖粗类别，却未覆盖细类别。建议引入有限、明确的 **region–fact** 标注集合 `F_k`，先判断每个事实是否 visually supported，再判断 caption 是否 entail 该事实。将 omission、contradiction、ambiguous 单独标注，不能把 contradiction 算普通未提及。

可用操作性定义：`C*(r_k|T) = Σ_f w_f G(f,I)[1-E(f,T)] / Σ_f w_f G(f,I)`，只在分母非零且事实可判定时评分。G 是视觉支持，E 是文本蕴含，权重预先固定。这是相对于标注事实集合的覆盖率，不是全部视觉信息，也不是 Shannon 信息量。unknown 用 mask 排除。课程第一阶段可限制到 entity identity，并明确这是更窄的任务。

### P0：区分内容预测与模型解释

训练一个外接 MLP 预测遗漏语义，不自动构成对 CLIP 内部决策的 faithful explanation。如果主要证据是人类语义标签，题目应定位为 semantic coverage/complementarity localization；只有同时证明输出对应被解释模型对独立任务的行为变化，才加强 interpretability claim。不要用同一个 CLIP 同时产标签、训练、评分再宣称发现了真实信息。

### P0：处理伪标签与泄漏

Flickr30K Entities 是 phrase grounding 标注，不是所有视觉事实的穷尽标注。找不到 phrase 不等于确定没有覆盖；同义表达、共指、关系、caption 间不同视角均会造成误标。删除某一 mention 后，句子可能仍通过别的 mention 表达目标。natural multi-caption pair 也不天然只改变目标，不能无审核地用于 locality。

所有同图 caption、crop、intervention 必须在同一 split；模板和类别组合留出另做泛化集，不要把“官方划分”误写成自带模板划分。过滤 `nobndbox`、scene-only、坏框、同一实体多 mention、多框、重叠/包含区域、关系依赖。不要将目标 phrase、被删 span、target ID 输入推理模型；这些只能用于训练监督/评估。

### P0：修复 locality 与指标

`j != k` 不能直接视为 unrelated：person 与 shirt 框重叠，关系涉及两个实体，多个框可能属于同一 chain。用预定义依赖掩码 `U(k)` 选择真的无关区域；无可用区域时不计算 locality，避免 `K-1=0`。

TIG 大不代表绝对 complementarity 正确，纯文本长度规则也能产生正 TIG；恒定分数则有完美 stability。需同时报告：

- 正向干预成功率、TIG、无关区域变化，固定验证集校准分数范围。
- 局部对比差 `D = Δ_target - mean(Δ_unrelated)`，同时保留绝对 off-target drift。
- **按 Δ 排序找被干预区域**的 Acc@1/MRR；不能混同于按 `q(T-)` 排序找所有未覆盖区域。
- 对独立 omission 标签计算 AP/AUROC、分层性能；多个未覆盖区域不能强制只有一个正确框。
- 有人工 mask 才报告 mask IoU；box proxy 的 IoU 不能等同真实区域 mask 质量。
- 以 image 为独立单位 bootstrap，并报告 3–5 个训练 seed。不要把同图数十个 edits 当独立样本扩大显著性。

### P0：补强基线

除了 saliency，应有直接解决同一任务的基线：检测/region caption → 文本蕴含/coverage；phrase-level max matching；同样训练数据和参数预算的 omitted-entity classifier；text-only、image-only、text-length、shuffled-image、constant。oracle phrase/GT fact coverage 单列为有额外监督的上界，不能伪装成普通基线。

Saliency 解释的是文本匹配证据，直接被 complementarity 模型击败不足以说明方法创新。其分数方向、归一化和 mask 预算应在 validation 固定。CIEA 的 region adaptation 如有做，要明确偏离原任务，不称原论文完整复现。

### P1：技术细节与可执行性

先投影 `v'=P_v v, t'=P_t t` 再计算所有 `v'⊙t'` 和 `|v'-t'|`；原式只给差项投影，维度不同时乘积仍无定义。明确 crop padding、resize、框坐标、region pooling、frozen encoder 版本、缓存 fingerprint、优化器、超参选择范围。补写 `L_ctrl` 的精确定义及控制对构建，不只留一个名字。

训练仅排名 loss 无法识别绝对 q 的含义；如果要把 q 当概率，应加独立 coverage 监督并校准，否则称 score。预注册主指标和主模型，避免反复在 test 上选择实验方向。

“Deletion faithfulness”须定义 recoverability：由谁从什么输入、以什么候选/问答形式恢复什么事实。按面积匹配 random/importance controls，加入 blur/mean-fill 等多种扰动，控制 mask 边缘、背景和 OOD 效应。文本已经含答案的 T+ 可作证据需求降低的对照，但不能和 T- 的恢复难度混为一谈。

## 3. 课程项目的合理收缩

必须交付：Flickr30K 小范围受控实体实验、inverse CLIP/phrase matching/shortcut controls、一个轻量 scorer、loss ablations、独立测试、错误分析、Agent Log。先做 200 train / 50 validation / 100 test images 的 pilot；这是可行性估计，非确定最终样本量。

有余力再做：一个 attribution baseline、natural-caption audit、较大 Flickr 子集。Visual Genome transfer、自动 proposals、SigLIP、金融分别升级为 stretch goals。不要同时把全部作为课程最低要求。

## 4. 分阶段执行与停止条件

1. **资源与数据**：检查 MPS、缓存模型、访问渠道；保存数据出处/版本和 split IDs。先跑 16–32 图吞吐与内存观察。无法得到真实图像时明确阻塞，不能用合成数据宣称方向成立。
2. **数据构造**：固定随机种子按官方 image split 取样；以实体级、无重叠目标/控制为首轮范围。保留编辑前后文本和排除原因。抽查 50–100 对后，人工错误率目标 <5%；当前自动过滤不替代人工审计。
3. **零训练基线**：inverse cosine、phrase max、length、area、constant、shuffled visual features。先查是否需要学习，及 label 本身是否可信。
4. **轻量学习**：冻结 CLIP，缓存 crops/text features；rank、rank+locality、rank+locality+control；公平加入监督的 coverage scorer。validation 选模型，test 只作一次阶段评估。
5. **判定**：必须优于最强任务直接基线，同时显著优于 text-only/shuffled，并通过自然改写/人工样本。建议把 +5 percentage points 的 primary rank accuracy 作为内部值得扩展的幅度参考，同时 paired image-bootstrap CI 应排除 0；这是投资决策门槛，不是录用门槛。若 pilot CI 太宽，先扩大样本，不能把不显著当成等效。
6. **扩展**：满足前述证据再进入独立 CVPR proposal；只在 synthetic edits 好看时暂停扩展，先修数据/定义。baseline 已接近上界时，转向更细的属性关系或有实际用途的预算信息补全。

## 5. 本机资源与计划边界

实测环境：Apple M1 Pro，8 CPU cores / 14 GPU cores，16 GB unified memory；工作盘约 822 GiB 可用；Python 3.12.3、PyTorch 2.7.0、Transformers 4.51.3；MPS available；本地 `openai/clip-vit-base-patch16` 可加载。适合 batch 8–16 的 frozen features 和小型 MLP。训练多模态大模型和全量多 backbone attribution 不作为本机首轮目标。实际时间以 pilot 测量为准，不提前承诺全量完成时间。

数据访问：官方 Flickr30K 图像页面要求申请，Entities annotation 公开可下载；公开 HF 镜像可供当前非商业课程研究获取小样本，需保留原始图像权利/用途约束，不重新分发图像。优先按 HTTP range 提取小样本，避免下载完整约 4.39 GB 压缩包。

结果、未完成项和复现命令另记 `pilot/README.md` 与 `pilot/results/`。goal 已建立；只有本轮评审、两级计划和可行范围的 pilot 被实际交付后才完成，未来 CVPR 研究仍按新 proposal 执行。

## 6. 文献增补与核查（2026-09-26）

- [CIEA, ACL 2025](https://aclanthology.org/2025.acl-long.1073/)：已有图文互补特征提取与检索，限制“互补信息”创新表述。
- [Zur et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1125/)：caption/description 区分，定位动机而非空间覆盖标注。
- [CompreCap, CVPR 2025](https://arxiv.org/abs/2412.08614)：对象、属性、关系的区域/场景图覆盖评估，是原稿缺失的重要邻近工作。
- [CapProbe, arXiv 2026](https://arxiv.org/abs/2608.11074)：区域对齐 QA 检查 caption factual coverage；直接削弱“首次空间化未覆盖语义”的宽泛主张。不能把“学习一个 map”作为唯一差异。
- [Detail caption benchmark](https://arxiv.org/abs/2405.19092)：补充 detailed caption evaluation 文献线。
- [CPI](https://arxiv.org/abs/2605.22651)：phrase intervention 用于数据选择，干预本身不是新概念。
- [Visual Credit Audit v2](https://arxiv.org/abs/2607.27069v2)：区分额外视觉支持与关系响应，不能直接把本任务称为因果分解。v1 HTML 标题不同，应锁定 v2。
- [Grad-ECLIP](https://arxiv.org/abs/2502.18816)：arXiv 当前元数据支持原稿 TPAMI 2026 与 DOI；DOI 直接网页本次未成功打开，不应据此判假。
- [CCI](https://arxiv.org/abs/2511.12978)：论文存在；原稿 CVF 2026 URL 本次打不开，最终提交前再次核查具体 proceedings metadata。
- [Flickr30K Entities 官方代码/标注](https://github.com/BryanPlummer/flickr30k_entities)：共指、多框、scene/no-box、官方 split 的依据。

这次是针对关键邻近工作的定向检索与方法核对，并非穷尽系统综述；novelty 结论应随完整文献比较更新。
