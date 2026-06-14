# 崔继轩 - AI 算法工程师项目汇报

您好！我是崔继轩。本仓库提炼了我在简历中向您汇报的，真实工业/医疗场景下的核心 AI 算法项目，重点展示了在**智能体搭建、基座模型微调、联邦学习、扩散模型**等方向的工程实践。

---
## 1. 基于 LangGraph 的工业基座模型微调 Agent 系统
> **关键词：** LangGraph + RAG + SQLite | LoRA微调 | Agent 搭建

**项目背景：** 针对工业模型训练中算法选型依赖人工、流程割裂与状态难追踪等问题，开发面向本地微调的对话式 Agent 系统，打通意图解析、计划生成、任务执行、结果展示与计划迭代闭环。
**核心亮点：**
* **对话式任务编排（理解、检索、规划）**：基于 LangGraph 构建 10 类命令路由，覆盖问答、RAG 检索、计划生成/修改、执行/取消、进度与结果查询；结合本地知识库与策略推荐模块，根据自然语言需求设计针对具体任务的特性分析、策略推荐、计划生成。
* **迭代式训练优化（记忆、执行、反馈）**：以 SQLite 构建任务状态记忆，持久化计划、运行、结果与错误信息；封装数据画像、配置生成、模型构建、训练推理、评测报告等工具，由 Worker 执行长时任务，并回传进度与失败诊断，支撑模型训练与优化的自动闭环。



<img width="1881" height="786" alt="图1-1" src="https://github.com/user-attachments/assets/6abd210a-1db4-4e9b-a4e4-9c297bffdea6" />

*图  1-1：系统架构图*

<br>

<video width="640" controls>
  <source src="https://github.com/CuiJixuan/JixuanCui-ProjectInfo/raw/main/%E6%BC%94%E7%A4%BA1-1.mp4" type="video/mp4">
  你的浏览器不支持视频播放，请点击链接查看：https://github.com/CuiJixuan/JixuanCui-ProjectInfo/blob/main/%E6%BC%94%E7%A4%BA1-1.mp4
</video>


*演示 1-1： Agent实机演示视频,支持数据分析、模型训练、总结分析以及普通闲聊*



<br>
<br>

---



## 2. 基于扩散生成模型的医疗重症辅助诊断桌面系统
> **关键词：** 扩散生成模型 (DDPM) | Tauri+Vue3 原生调用 | ONNX 毫秒级极简推理

**项目背景：** 面向安医大一附院真实临床 ARDS 早期预警，解决医疗数据极度稀缺、类别严重不平衡及老旧设备无网/无算力的痛点。
**核心亮点：**
* **特征掩码扩散生成与阈值动态寻优：** 提出特征掩码扩散模型FM-DDPM，训练时随机掩码特征维度并引入辅助自监督重建损失以学习特征间内在关联，约束生成空间以扩充非平衡训练数据集；结合验证集F1分数最大化阈值寻优，在12种分类模型上评估，相较于SMOTE算法F1与ACC提升18.9%与3.8%，相较于DDPM算法F1提升3.3%与0.6%，最优分类器F1与ACC为80.1%与94.9%。
* **推理管线解耦与系统原生调用：** 彻底剥离 Python 依赖，采用“ONNX-JS-ONNX”三明治架构实现实时特征增广；基于 Tauri 提权静默抓取剪贴板 28 项生化指标完成毫秒级推理，最终交付仅十几 MB 的免安装 exe 看板，完美适配医院老旧无网设备。



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
## 3. 基于联邦学习的分布式多模态基座模型微调平台
> **关键词：** 云边协同架构 | 基座模型+LoRA | 全场景异构适配算法 | 全链路跨网段联调

**项目背景：** 依托工信部创新发展工程项目（1.75亿），解决工业边缘侧数据孤岛、多维异构（分布/算力/内存/带宽）及隐私安全痛点。
**核心亮点：**
* **架构解耦与算法集成：** 主导设计底层网关与算法解耦架构。集成 INT8+LoRA 降低算力门槛，支持 SFL 分割联邦极致压缩带宽开销，结合 LDP 构筑安全壁垒。
* **全场景异构感知：** 针对模态、数据、内存、算力四维异构，定制图信息瓶颈 GIB、双轨近端正则、秩异构 LoRA 蒸馏及异步聚合机制。
* **从 0 到 1 跨网段交付：** 引入虚拟覆盖网络穿透隔离，基于 Docker 与无状态 gRPC 网关实现云边解耦与断点容灾。


<img width="2343" height="1056" alt="image" src="https://github.com/user-attachments/assets/eca7a212-37ec-4b16-a8d1-f557fed473f1" />


*图 3-1：系统架构图*

<br>

<img width="2274" height="963" alt="1-2" src="https://github.com/user-attachments/assets/1eb12ae6-e39b-459a-a3b7-0d452842721f" />


*图 3-2：服务端控制台*

<br>

<img width="1740" height="504" alt="1-3" src="https://github.com/user-attachments/assets/4eb4b2ae-7c40-4185-a8c2-ff990f75d5ba" />


*图 3-3：全链路监测大屏（展示多用户并发下的测试准确率）*

<br>

<img width="1770" height="591" alt="1-4" src="https://github.com/user-attachments/assets/a637e776-deb8-40a2-b21d-b71f16051338" />


*图 3-4：用户端大屏*

<br>
<br>

---

📫 **联系方式:** jixuancui@njust.edu.cn | 📱 13966116170
