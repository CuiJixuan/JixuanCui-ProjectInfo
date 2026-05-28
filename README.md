# 崔继轩 - AI 算法工程师项目汇报

您好！我是崔继轩。本仓库提炼了我在简历中向您汇报的，真实工业/医疗场景下的核心 AI 算法项目，重点展示了在**基座模型微调、多模态异构联邦、扩散模型数据增广及轻量化离线部署**等方向的可视化成果与工程实践。

---

## 1. 基于联邦学习的分布式多模态基座模型微调平台
> **关键词：** 云边协同架构 | 基座模型+LoRA | 全场景异构适配算法 | 全链路跨网段联调

**项目背景：** 依托工信部创新发展工程项目（1.75亿），解决工业边缘侧数据孤岛、多维异构（分布/算力/内存/带宽）及隐私安全痛点。
**核心亮点：**
* **架构解耦与算法集成：** 主导设计底层网关与算法解耦架构。集成 INT8+LoRA 降低算力门槛，支持 SFL 分割联邦极致压缩带宽开销，结合 LDP 构筑安全壁垒。
* **全场景异构感知：** 针对模态、数据、内存、算力四维异构，定制图信息瓶颈 GIB、双轨近端正则、秩异构 LoRA 蒸馏及异步聚合机制。
* **从 0 到 1 跨网段交付：** 引入虚拟覆盖网络穿透隔离，基于 Docker 与无状态 gRPC 网关实现云边解耦与断点容灾。


<img width="2355" height="1059" alt="1-1" src="https://github.com/user-attachments/assets/c9d2d735-d4c3-4710-8dcf-a60dbb9e2d7b" />

*图 1-1：系统架构图*

<br>

<img width="2274" height="963" alt="1-2" src="https://github.com/user-attachments/assets/1eb12ae6-e39b-459a-a3b7-0d452842721f" />


*图 1-2：服务端控制台*

<br>

<img width="1740" height="504" alt="1-3" src="https://github.com/user-attachments/assets/4eb4b2ae-7c40-4185-a8c2-ff990f75d5ba" />


*图 1-3：全链路监测大屏（展示多用户并发下的测试准确率）*

<br>

<img width="1770" height="591" alt="1-4" src="https://github.com/user-attachments/assets/a637e776-deb8-40a2-b21d-b71f16051338" />


*图 1-4：用户端大屏*

<br>
<br>

---

## 2. 基于扩散生成模型的医疗重症辅助诊断桌面系统
> **关键词：** 扩散生成模型 (DDPM) | Tauri+Vue3 原生调用 | ONNX 毫秒级极简推理

**项目背景：** 面向安医大一附院真实临床 ARDS 早期预警，解决医疗数据极度稀缺、类别严重不平衡及老旧设备无网/无算力的痛点。
**核心亮点：**
* **特征掩码扩散生成与阈值动态寻优：** 提出特征掩码扩散模型FM-DDPM，训练时随机掩码特征维度并引入辅助自监督重建损失以学习特征间内在关联，约束生成空间以扩充非平衡训练数据集；结合验证集F1分数最大化阈值寻优，在12种分类模型上评估，相较于SMOTE算法F1与ACC提升18.9%与3.8%，相较于DDPM算法F1提升3.3%与0.6%，最优分类器F1与ACC为80.1%与94.9%。
* **推理管线解耦与系统原生调用：** 彻底剥离 Python 依赖，采用“ONNX-JS-ONNX”三明治架构实现实时特征增广；基于 Tauri 提权静默抓取剪贴板 28 项生化指标完成毫秒级推理，最终交付仅十几 MB 的免安装 exe 看板，完美适配医院老旧无网设备。

*(以下为系统极速推理交互演示)*


<img width="1450" height="747" alt="image" src="https://github.com/user-attachments/assets/cf9c3b59-8e32-4b0f-889c-cd6ce4e89431" />

*图 2-1：项目流程图*

<br>


<img width="1728" height="890" alt="image" src="https://github.com/user-attachments/assets/698deeba-8631-4be9-a82a-d8bda886dbf7" />

*图 2-2：算法示意图*

<br>


https://github.com/user-attachments/assets/cffe03d4-1f82-4ce4-a852-e5aad1fee06d

*演示 1：基于 Tauri 提权静默抓取 28 项生化指标，完成毫无卡顿的毫秒级无感推理*

<br>

<img width="1602" height="673" alt="image" src="https://github.com/user-attachments/assets/c349cc39-b20a-4dca-baeb-eca52fea8c0e" />

*图 2-3：相较于SMOTE算法F1与ACC提升18.9%与3.8%，相较于DDPM算法F1提升3.3%与0.6%，最优分类器F1与ACC为80.1%与94.9%*

<br>
<br>

---

## 3. 基于视觉基座模型微调的复杂时变工况缺陷分类系统
> **关键词：** RsLoRA 高秩抗噪 | 重参数化模型折叠 | ONNX 跨端部署

**项目背景：** 面向恶劣工业时变工况（噪声、模糊、光照畸变交织），突破传统微调的高秩退化瓶颈，并实现厂区老旧工控机免环境部署。
**核心亮点：**
* **双轨抗噪优化（结构扩充 & 方差稳定）**：针对传统 LoRA 在干扰下性能坍塌与高秩时信号衰减等问题，提出并行双旁路架构以扩充模型在多维时变干扰下的表征容量，配合修正缩放因子稳定特征方差以产生平滑正则化效应抵抗噪声。在小样本+混合干扰下，Rank=64时性能超越开销更大的Adapter等算法，Rank=256 时准确率达 84.67%，较传统 LoRA 绝对提升 16.67%。
* **极简离线部署（重参数化 & ONNX）**：利用重参数化将 LoRA 增量矩阵无损折叠至主干网络，实现推理端“0参数膨胀与0延迟增长”；将计算图静态化编译为 ONNX 格式彻底剥离 Python 依赖，以极小体积在老旧厂区终端上实现跨 CPU/GPU 的低资源离线推理。


<img width="1467" height="728" alt="image" src="https://github.com/user-attachments/assets/ca989b84-085a-4079-a8a8-b7136a9104c3" />


*图  3-1：算法流程图*

<br>

<img width="1793" height="767" alt="image" src="https://github.com/user-attachments/assets/f0d0e6ed-1ef7-40a7-bb69-790ac6e42df5" />


*图 3-2： 无干扰工况测试，极低开销逼近全量微调上限（93.00% vs. 94.33%）, 超越传统LoRA（85.33%）以及开销更高的Adapter（91.33%）*

<br>

<img width="1778" height="755" alt="image" src="https://github.com/user-attachments/assets/ea372b1f-07e1-4a7d-990b-293b141c1fbf" />

*图  3-3：混合干扰工况测试，Rank=64时性能超越开销更大的Adapter等算法，Rank=256 时准确率达 84.67%，较传统 LoRA 绝对提升 16.67%*

<br>
<br>

---
📫 **联系方式:** jixuancui@njust.edu.cn | 📱 13966116170
