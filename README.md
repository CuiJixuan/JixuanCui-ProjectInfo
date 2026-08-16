# 崔继轩｜AI 算法工程师项目汇报

您好，我是崔继轩。本仓库汇总了我在真实工业与医疗场景中的核心 AI 项目，重点展示在**多智能体系统、基座模型高效微调、扩散生成模型、联邦学习与云边协同**等方向的算法研究与工程实践。

---

## 1. 基于 LangGraph 的工业模型开发 Agent 系统

> **关键词：** Multi-Agent｜LangGraph｜ReAct｜RAG｜Agent Skills｜MCP

**项目概况：** 面向本地工业模型开发，构建对话式多智能体系统，实现需求理解、知识检索、实验执行与迭代复盘闭环。

**项目产出：** Agent 软件系统 1 套，含 8 个 Agent Skills 及 100 余个本地及 MCP 工具。

### 核心亮点

* **多智能体编排与 ReAct 决策（LangGraph）：**基于 LangGraph 构建“Supervisor + Experts”多智能体架构，将知识调研、常规办公、算法实现、实验执行和独立审查拆分为 5 类专家。Supervisor 负责任务拆解、顺序或并行委派及结果汇总；Experts 在独立上下文和工具白名单内运行局部 ReAct，通过“决策—执行—观察”循环推进任务。进一步设计结构化实验上下文补全、执行权限预检、调用预算和重复调用检测等机制，减少工具冲突、上下文污染与无效循环。

* **分层记忆与工业知识增强（SQLite + RAG）：**按照信息生命周期划分单轮记忆、会话记忆和长期记忆，分别保存 ReAct 轨迹、多轮任务状态与跨会话可复用经验；将数据画像、实验规格和任务状态结构化保存并绑定当前会话，避免上下文歧义和历史结果误用。围绕工业模型开发构建可追溯知识库，结合本地向量检索与字段检索召回文献依据、历史实验和相似案例，为算法选择、参数推荐、失败诊断与结果复盘提供支持。

* **工业 Skills 与实验闭环（Agent Skills + MCP）：**围绕工业模型开发构建 8 个项目 Skills，覆盖资源发现、组件添加、数据诊断、实验构建、方案探索和结果报告，统一约束触发条件、结构化输入、工具序列、完成标准与异常恢复。通过 Function Calling 编排约 100 个本地 Tool 及外接 MCP Tool；训练任务由 Worker 异步执行并持续回写指标与日志，当结果未达到用户设定目标时，系统根据实验观察进行诊断、修复或运行新候选方案，形成“数据诊断—组件准备—实验执行—结果审查—迭代探索”闭环。

<!-- 图 1-1 如仍为旧版单 Agent 架构，请替换为最新 Supervisor + Experts 多智能体架构图。 -->


<img width="1273" height="651" alt="image" src="https://github.com/user-attachments/assets/04e06423-023b-4076-ac96-0ae2ae5d3605" />


*图 1-1：工业模型开发 Agent 系统架构图*

<br>


<img width="1420" height="747" alt="image" src="https://github.com/user-attachments/assets/3be39ee8-bbac-4bfa-8c5c-7087e79b7e22" />


*图 1-2：工业 Agent Skills*

<br>

<img width="1884" height="630" alt="通用问答测试" src="https://github.com/user-attachments/assets/08b15668-26b4-4822-8e48-4fcefda18ff5" />

*图 1-3：通用问答测试——系统能够识别非任务型请求并完成正常交互*

<br>

<img width="1872" height="573" alt="工业数据诊断" src="https://github.com/user-attachments/assets/2519e5e7-c61e-4b90-adef-e78a40372851" />

*图 1-4：数据诊断——分析产线数据规模、类别分布、噪声与潜在风险*

<br>

<img width="1835" height="678" alt="模型训练任务" src="https://github.com/user-attachments/assets/df63b1be-7aab-4608-b2f0-29b3d3a469b5" />

*图 1-5：实验执行——根据数据特性生成方案、调用工具并完成模型训练*

<br>

<img width="1428" height="471" alt="实验结果分析" src="https://github.com/user-attachments/assets/7f9e4c8f-3c1a-41f7-a010-cee9c94ac9aa" />

*图 1-6：结果复盘——分析实验结果并生成下一轮优化方向*

<br>

https://github.com/user-attachments/assets/1762e5ff-44d5-4aa9-9869-34c6440bbfbb

*演示 1-1：Agent 系统实机演示，支持数据诊断、实验执行、结果分析与通用问答*

<br>
<br>

---

## 2. 基于视觉基座模型微调的复杂时变工况缺陷分类系统

> **关键词：** TCD-LoRA｜PEFT｜Robust Training｜ONNX｜INT8 Quantization

**项目概况：** 针对工业缺陷分类中的小样本与多维干扰问题，提出抗干扰参数高效微调方法，并完成重参数化与 ONNX 部署。

**项目产出：** TCD-LoRA 算法模型 1 套，Atlas NPU 实板部署原型 1 套。

### 核心亮点

* **抗扰参数高效微调（TCD-LoRA）：**提出 Task-Corruption Decoupling LoRA，在 Attention 的 Q/V 层并联任务 LoRA 专家与仅在训练阶段启用的扰动 LoRA 专家，采用纯净—受扰双视图联合训练。通过预测一致性约束稳定分类边界，并通过低秩子空间正交约束解耦任务语义与扰动语义。在 7 类受扰工况的平均指标上，Rank=32/64/256 均优于其余 8 种 LoRA 方法；Rank=256 时准确率达到 90.52%，仅使用全量微调 28.62% 的可训练参数量，接近全量微调 91.57% 的准确率，较同秩 LoRA 和 RsLoRA 分别提升 13.42 和 2.38 个百分点。

* **重参数化离线部署（ONNX + INT8）：**推理前移除扰动专家，并将任务 LoRA 增量无损合并至主干权重；完成 PyTorch/ONNX 数值对齐、Dynamic INT8 与 Static INT8 QDQ 量化，以及 Windows x86 CPU 上的延迟、吞吐和内存测试。Dynamic INT8 将模型由 107.77 MiB 压缩至 29.98 MiB，体积减少 72.18%；Rank=32/64/256 的准确率仅下降 1.00/0.00/0.67 个百分点，batch=8 时吞吐量较 FP32 提升 17.97%，RSS 增量降低 30.18%。

<!--
原图 2-1 至图 2-3 对应旧版 RepRsLoRA 方法及旧实验指标。
建议替换为：
1. TCD-LoRA 双专家与双视图训练架构图；
2. 7 类受扰工况下与 8 种 LoRA 方法的对比结果；
3. ONNX、Dynamic INT8、Static INT8 QDQ 的部署测试结果。
-->

<img width="1588" height="690" alt="image" src="https://github.com/user-attachments/assets/0c830ef2-67bd-437c-b68d-bb9289735d1f" />


*图 2-1：TCD-LoRA 双专家抗扰微调架构*

<br>

<img width="1007" height="669" alt="image" src="https://github.com/user-attachments/assets/a12daee6-980d-452a-841b-f1191d22f1e1" />


*图 2-2：不同 Rank 下 TCD-LoRA 与参数高效微调方法的受扰工况对比*

<br>

<img width="803" height="257" alt="image" src="https://github.com/user-attachments/assets/06b7d504-596b-43b6-b662-7f976599e73f" />

*图 2-3：ONNX 数值对齐、模型压缩及 Windows x86 CPU 推理测试结果*

<br>

<img width="875" height="556" alt="image" src="https://github.com/user-attachments/assets/eb4f81fc-5c65-4940-b50e-8446bc534d1e" />


*图 2-4：延迟测试*

<br>

<img width="955" height="633" alt="image" src="https://github.com/user-attachments/assets/f26c1423-dbd5-4dc1-84b0-5537fd8cb18b" />


*图 2-5：吞吐测试*

<br>


<img width="997" height="633" alt="image" src="https://github.com/user-attachments/assets/bf700670-0dc9-4fa2-b475-f0b4a89a7a8b" />


*图 2-6：内存显存增量测试*

<br>


<br>

---

## 3. 基于扩散生成模型的医疗重症辅助诊断桌面系统

> **关键词：** FM-DDPM｜Clinical Feature Engineering｜ONNX Runtime｜Vue 3｜Tauri 2

**项目概况：** 依托安医大一附院真实 ARDS 数据，使用扩散模型增强非平衡数据集，并交付离线辅助诊断桌面软件。

**项目产出：** SCI 中科院二区论文 1 篇，桌面诊断软件 1 套。

### 核心亮点

* **数据清洗与临床特征工程：**围绕 28 项生命体征与生化指标构建可复现的数据处理管线，完成异常值逻辑校验、缺失值补齐和标准化处理；进一步构造 NLR、PLR、MAP 等临床衍生特征，并根据相关系数阈值剔除冗余维度，最终形成统一的 30 维结构化输入，用于生成增强、分类训练与 ONNX 推理。

* **掩码扩散生成与阈值动态寻优：**提出特征掩码扩散模型 FM-DDPM，在训练过程中随机掩码部分特征维度，并引入辅助自监督重建损失，学习临床指标之间的内在关联，约束生成空间并增强少数类样本。结合验证集 F1 最大化进行分类阈值寻优，在 12 种分类模型上评估；相较 SMOTE，F1 与 ACC 分别提升 18.9% 和 3.8%；相较 DDPM，F1 与 ACC 分别提升 3.3% 和 0.6%；最优分类器 F1 与 ACC 达到 80.1% 和 94.9%。

* **离线推理管线与桌面终端交付：**面向医院内网隔离和终端老旧等部署约束，将“缺失补齐—特征衍生—标准化—分类预测”全链路固化为 ONNX 静态计算图；基于 Vue 3 和 Tauri 2 开发离线桌面应用，支持 28 项生命体征与生化指标录入、风险预测与本地毫秒级推理，最终交付约 20 MB 的免安装桌面软件。

<img width="1450" height="747" alt="医疗重症辅助诊断系统流程图" src="https://github.com/user-attachments/assets/cf9c3b59-8e32-4b0f-889c-cd6ce4e89431" />

*图 3-1：医疗重症辅助诊断系统流程图*

<br>

<img width="1728" height="890" alt="FM-DDPM 算法示意图" src="https://github.com/user-attachments/assets/698deeba-8631-4be9-a82a-d8bda886dbf7" />

*图 3-2：FM-DDPM 特征掩码扩散生成方法*

<br>

https://github.com/user-attachments/assets/cffe03d4-1f82-4ce4-a852-e5aad1fee06d

*演示 3-1：基于 Tauri 的离线桌面软件，实现临床指标录入与本地毫秒级推理*

<br>

<img width="1602" height="673" alt="医疗重症辅助诊断实验结果" src="https://github.com/user-attachments/assets/c349cc39-b20a-4dca-baeb-eca52fea8c0e" />

*图 3-3：FM-DDPM 与 SMOTE、DDPM 等数据增强方法的分类性能对比*

<br>
<br>

---

## 4. 基于联邦学习的分布式多模态基座模型微调平台

> **关键词：** Federated Learning｜Multimodal Foundation Model｜LoRA｜Cloud-Edge Collaboration｜Docker｜gRPC

**项目概况：** 依托工信部创新发展工程项目，面向工业边缘侧数据与资源受限问题，研发云边协同的分布式联邦学习平台。

**项目产出：** SCI 中科院二区论文 2 篇，普刊论文 1 篇，授权发明专利 3 件，联邦学习系统平台 1 套。

### 核心亮点

* **平台通用设计与算法集成：**主导设计底层通信网关与上层联邦算法解耦的系统架构，支持主流联邦学习方案接入。针对数据稀缺问题，引入多模态基座模型提供先验知识；针对算力受限问题，采用 INT8 量化与 LoRA 微调降低训练开销；针对带宽受限问题，集成分割联邦学习与梯度稀疏化；针对投毒攻击和隐私风险，结合本地差分隐私与动态梯度裁剪增强系统安全性。

* **边端全场景异构感知适配：**面向模态、数据、内存与算力异构构建自适应优化机制。通过图信息瓶颈 GIB 过滤多模态结构冗余，通过双轨近端正则与特征对抗缓解用户数据分布差异，通过秩异构 LoRA 蒸馏与免填充聚合支持不同内存条件下的模型训练，并通过异步聚合降低慢节点对整体训练效率的影响。

* **跨网段联调与原型机交付：**负责平台从 0 到 1 的核心功能验证，引入虚拟覆盖网络解决跨网段通信问题，并基于 Docker 实现云边模块容器化交付；构建轻量级无状态 gRPC 通信网关，利用 Redis 与 MinIO 的 Claim-Check 模式完成任务调度、模型文件传输和断点续传；集成 Prometheus、Grafana 与 Jaeger，实现训练指标、系统资源与调用链路的全流程监控。

<img width="2343" height="1056" alt="联邦学习平台系统架构图" src="https://github.com/user-attachments/assets/eca7a212-37ec-4b16-a8d1-f557fed473f1" />

*图 4-1：云边协同联邦学习平台系统架构*

<br>

<img width="2274" height="963" alt="联邦学习平台服务端控制台" src="https://github.com/user-attachments/assets/1eb12ae6-e39b-459a-a3b7-0d452842721f" />

*图 4-2：联邦学习平台服务端控制台*

<br>

<img width="1740" height="504" alt="全链路训练监控大屏" src="https://github.com/user-attachments/assets/4eb4b2ae-7c40-4185-a8c2-ff990f75d5ba" />

*图 4-3：全链路训练监控大屏，展示多客户端并发训练与测试准确率*

<br>

<img width="1770" height="591" alt="联邦学习平台客户端界面" src="https://github.com/user-attachments/assets/a637e776-deb8-40a2-b21d-b71f16051338" />

*图 4-4：联邦学习平台客户端界面*

<br>
<br>

---

📫 **联系方式：** [jixuancui@njust.edu.cn](mailto:jixuancui@njust.edu.cn) ｜ 📱 13966116170
