# 当前进度与验收

## 2026-10-07 更新：U-Net 最终 50 场评测完成

- 在固定 50 个开发场景上比较 5k–40k 共 8 个 EMA 快照（400 场）。按 >87% 成功数、平均最大覆盖率、较早步数排序，选定 step40000；开发成绩 43/50、平均最大覆盖率 86.3927%。
- 锁定检查点后独立生成最终 50 场，generation seed=2802778101；采用用户要求的 `user_requested_final_50_v1` 协议修订，原 100 场默认配置不冒充已完成。
- 最终结果：>87% 为 **38/50（76%）**；>95% 为 **31/50（62%）**；平均最大/终态覆盖率 76.09%/70.23%；L4 单进程规划延迟 p50/p95 为181.6/189.4ms。
- 证据入口：[最终验收](reports/unet-l4-20261007/acceptance.md)、[逐场CSV](reports/unet-l4-20261007/per-scene.csv)、冻结场景、选模锁、完整轨迹与全部50场动作回放。模型及视频在 README 链接的 Release 提供。
- 最佳检查点 SHA256：`6d8e1fcb0342f377e6973be82139b63302feef64f0cf037f6db9b75da13873b4`。
- 固定 Pymunk 6.11.1 兼容 gym-pusht 0.1.6；环境包装层补充真实初始 coverage，避免 reset 信息缺失时被计为0。真实观察与物理步进一致性检查通过；未改用户模型/训练核心，也未新增 mask 防护。
- 本次评测、选择、报告及场景相关检查共 **19 passed**；不代表旧全套测试已适配或通过。
- 全206条演示训练、单训练种子seed0；最终场景不是专家演示留出集，不能证明与训练姿态完全不重合。DiT与同协议ACT比较尚未完成。

## 2026-10-07 更新：Colab L4 正式训练已完成

- 用户已实现观测编码、U-Net、扩散损失、参数更新和 EMA。旧的“所有入口未实现”描述仅属于下方 9 月交付快照。
- CPU 真训练 100 步通过：batch64，前10步平均 loss 1.4208，后10步 0.1763；完整检查点保存与重载通过。证据：`runs/cpu-smoke-100-20261007/verification.json`。
- CPU 精确恢复通过：100→102 与 100→101→102 的权重、EMA、优化器、LR、sampler、RNG 逐项相等。证据：`runs/cpu-resume-20261007/verification.json`。
- 当前 U-Net 已通过推理适配器接入采样，真实检查点能生成有限 `[16,2]` 动作；尚无正式闭环成功率。
- 增加带中文注释的 `scripts/colab_train.py`、`scripts/colab_preflight.py` 和 `notebooks/PushT_Diffusion_UNet_L4_20261007.ipynb`。
- 正式协议：40,000 步、batch64、seed0、float32；每1,000步保存 last，每5,000步保留独立候选快照；开发场景选点延后执行。
- L4 + 8 个 DataLoader workers 的真实预检查已通过：100步训练、Drive校验和、精确恢复、小批拟合。固定16样本的独立噪声检查 loss 1.45885 → 0.09129（400步）。证据：`reports/colab-l4-20261007/preflight-workers8.json`。
- 20步同步测速平均 0.1029秒/步（数据等待0.00215秒、传输0.00614秒、更新0.09461秒），40k纯更新估计1.143小时，未计保存和评测。
- 用户报告页面中断后，重新连接 L4 并从 Drive 核实：`last.pt` 已完成40,000步，校验和与完整检查点结构验证通过，`status.json` 为 `complete`，无需补训。最终loss0.0042、近100步均值0.0069；原运行耗时4578.33秒。证据：`reports/colab-l4-20261007/completion-verification.json`。这证明训练及保存完成，闭环成功率仍待评测。
- Notebook：https://colab.research.google.com/drive/1idb6vRFXNuDWisXpInCqWjp17I31KRyJ
- Drive目录：`MyDrive/pusht-diffusion/20261007-unet-seed0/formal/`；运行时数据在本地/content，检查点与日志持久保存Drive。
- 最终8-worker源码和数据包：`pusht-diffusion-colab-20261007-workers8.zip`，云端SHA256 `118ea94039afc5e6d9babf47b752cd15c8db93f34561cd9355721a739d38b94b`；完整初始包另行保留。
- 旧 `tests/test_runtime.py` 引用已移除类，不能把历史全套测试结果当作当前全套通过；本次没有增加用户不希望的 mask 防护。

---

更新：2026-09-24。外围工程交付完成；新扩散模型与正式训练算法等待用户实现。没有进行训练、云端运行或正式扩散策略评测。

## 已实现并检查

- 独立包、固定配置、命令行、来源记录；生产包无旧工作区导入依赖。
- 全206回合数据审计：25,650帧，25,444有效动作起点。标量审计不解码图像；实际样本检查只解码4帧。
- 两帧相邻历史、首帧复制、尾部最后有效动作重复、布尔有效位、无跨回合窗口。原始记录最小最大值统计；外部审计不能绕过当前数据验证。
- 有界按需图像解码；外部标准化缓存验证路径有合成检查，本机未发现可用旧完整缓存。
- 20步显式噪声推理和100步诊断调度器；保留各子模块模式及随机状态；只有最终物理动作裁剪。
- 通用完整检查点容器、校验和、原子保存与目录镜像；已消费批次游标；全局与命名随机数状态；通用追加事件日志。
- 50个开发场景已冻结，继承45个旧记录来源的排除信息；最后100个场景仍未生成。最终生成会继承开发/调试文件中的历史排除台账。
- 策略无关逐步评测、87%越过后继续到95%或终止、指标重算、全部无序配对、Wilson区间、完整Markdown汇总表和回放视频。
- 现有 ACT（Action Chunking with Transformers，基于 Transformer 的动作分块）迁移验证：8个真实观测时刻，其中7组相邻历史图像不同，最大动作误差0。
- ACT真实12步闭环与保存动作回放；回放覆盖率误差0。它们仅是工程检查。
- 本地页面暂停/继续、同场景重放、新场景、干预与历史重建、未加载模型状态；控制和最新帧传输；同源检查和非对象消息处理；断线等待原生工作线程后关闭物理环境。
- 默认命令行服务记录独立交互调试台账，用于最终种子/姿态排除，不写正式评测结果。

## 最终证据

| 内容 | 文件 |
|---|---|
| 当前代码下重跑的数据、ACT比较、短闭环、环境版本和边界审计 | [engineering-final-v2/verification.json](reports/engineering-final-v2/verification.json) |
| 真实短闭环完整计划和逐步动作 | [engineering-final-v2/act-smoke.json](reports/engineering-final-v2/act-smoke.json) |
| 人读短闭环报告 | [engineering-final-v2/smoke-report.md](reports/engineering-final-v2/smoke-report.md) |
| 回放校验来源 | [engineering-final-v2/act-smoke.provenance.json](reports/engineering-final-v2/act-smoke.provenance.json) |
| 显式启用所有权审计后的全套结果 | [pytest-final.xml](reports/pytest-final.xml) |
| 全部19个受保护方法检查 | [ownership-final.json](reports/ownership-final.json) |
| 父任务实际浏览器验收 | [engineering/browser-review.md](reports/engineering/browser-review.md) |
| 生成图片、提示词与结构说明 | [architecture.md](docs/architecture.md) |

最终源代码指纹：`ea3231e35d3ebb213e77ef809510195dbd2ac0629030e223fdaf17b1782bf494`。

最终自动化命令：

```bash
PYTHONPATH=src:.deps PYTHONDONTWRITEBYTECODE=1 PUSHT_DELIVERY_AUDIT=1 \
/Users/marscolonizer/embodied-learning/wam-roadmap/mini-wam/.venv/bin/python \
-m pytest -q -p no:cacheprovider --junitxml=reports/pytest-final.xml
```

结果：**20 passed，3 skipped**。三个跳过是两种真实模型接口/重载和用户损失检查，因为其入口尚未实现。启用了交付占位所有权检查；日常默认运行会额外跳过这个交付专用检查，用户实现后不会因其失败而阻塞日常测试。

已保留三个依赖弃用警告，来自现有FastAPI/Starlette测试客户端和Pygame；没有为消除警告改动共享环境。实际环境运行还打印了旧依赖中重复多媒体动态库的警告，检查本身成功；本次未宣称重新建立纯净环境。

早期 `reports/engineering`、`reports/engineering-final` 是历史快照，修复前代码指纹不同；最终以 `engineering-final-v2` 为准。浏览器记录单独保留在父任务写入的位置。

## 本地服务

最终服务地址：`http://127.0.0.1:7861/live`。

交付时进程19414，工具会话49528，已加载最终断线清理实现并保持可用。进程编号是交付快照，之后重新启动会变化。启动命令见README；服务仅监听本机回环地址。

父任务实际浏览器观察了ACT运行超过22,500步、暂停时拖动物体而不推进步数、同场景重放和新场景、未实现骨干禁用。它们是交互工程证据，不是正式策略效果。最终重启只用于加载关闭清理修复。

## 等待用户实现

- 观测编码器、时间编码器、条件残差块、U-Net（U-shaped Network，U形网络）、DiT（Diffusion Transformer，扩散 Transformer）及组装类。
- 训练损失、反向传播、优化器、学习率、EMA（Exponential Moving Average，指数移动平均）更新、正式训练循环及工程工具接入。

这些方法仍抛出 `NotImplementedError`。没有在测试、范例或诊断脚本提供替代学习模型/训练器。

## 等待真实模型才能验证

- 模型与条件通路梯度、实际损失学习、小批拟合和随机噪声诊断。
- 真实优化器与平均参数的端到端断点恢复；云盘镜像持久性。
- 两个扩散骨干真实采样、闭环效果、显存、L4耗时与预算。
- 训练后开发选点、最后100新场景、正式三策略比较。

先运行 [学习路线](docs/learning.md) 中的数据窗口检查，再按接口逐模块实现。未来真实接口检查入口为 `tests/test_learner_contract.py`。
