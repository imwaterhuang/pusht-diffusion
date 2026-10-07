# 公共接口与状态约定

## 模型和推理

`LearnerDiffusionModel(config, initialize_pretrained=False)` 是检查点加载的固定构建入口。构造、`encode_observation(images,positions) -> [B,516]`、`denoise(noisy_actions,k,condition) -> [B,16,2]` 都由你实现。类继承 `torch.nn.Module`，参数遵循标准 `state_dict` 约定。

`DiffusionPolicy.predict_action_chunk(history, scene_seed=..., replan_index=...) -> numpy [16,2]` 是网页与评测共用入口。`history` 是最近两个原始环境观测，含 `pixels [96,96,3]` 与 `agent_pos [2]`。策略内部处理归一化、20步采样、反归一化与最终环境边界裁剪。`sample` 接受显式随机数生成器，可用于未来模型重载的一致性检查。

`make_scheduler` 仅是库调度器工厂；当前库版本 Diffusers 0.35.2。DDIM（Denoising Diffusion Implicit Models，去噪扩散隐式模型）为20步，DDPM（Denoising Diffusion Probabilistic Model，去噪扩散概率模型）诊断为100步。`clip_sample=False` 是本项目采样选择，唯一动作裁剪发生在最终物理坐标。

## 数据

`PushTDataset(root, image_cache=None, audit=None)` 默认重新审计206回合数据，即使传入审计对象也不跳过当前文件指纹、行顺序和回合范围检查。测试的小型合成数据可显式设置 `require_reference_counts=False`；真实实验使用默认值。

图像读取对象每进程维护最多256帧解码缓存；多个重叠窗口不会反复解码同一近期帧。大型内存映射缓存必须通过数据、标准化、形状、文件长度、校验和及源图像抽查。

## 通用恢复容器

`make_checkpoint` 接收调用方已经计算的 `raw`、`ema`、`optimizer`、`lr`、`sampler`、`rng` 状态，完全不做训练更新。容器携带结构、模型类型、完整配置及指纹、数据指纹、归一化、已完成步数、代码指纹和附加信息。`save_checkpoint` 原子保存并生成 `.pt.sha256.json`；可选 `mirror` 是已挂载目录。`load_checkpoint` 使用仅权重安全读取模式并验证元数据，用户恢复时应提供预期配置、数据和归一化。

`DeterministicBatchSampler` 生成确定性的批次顺序；预取只增加临时位置。你在成功更新后调用 `mark_consumed` 才推进持久化位置。恢复时销毁旧迭代器和数据加载器，再从已消费游标创建新迭代器，不能继续使用恢复前的预取队列。

`capture_training_rng(generators)` / `restore_training_rng(state,generators)` 包括 Python、NumPy、Torch 及可用加速器状态，并接受命名的训练噪声和数据加载器随机数生成器。RNG（Random Number Generator，随机数生成器）名集合和设备必须一致。将 DataLoader 的独立生成器也显式传入，避免恢复后创建迭代器时意外消耗训练噪声流。

`append_event(path,event)` 是通用追加日志接口，自动建目录、使用严格数据格式拒绝非有限数值。训练步数、损失、学习率、耗时等事件由你在正式循环中调用记录。一次训练运行应只有一个日志写入者。

此处测试证明容器、游标、镜像复制和随机数状态往返；**未证明真实优化器更新的端到端断点续训**。真实云盘持久性也未验证。

## 评测与场景锁定

50个开发场景已经冻结。加载时验证版本、分组、数量、哈希、唯一性和真实物理引擎的合法初始状态。最终100场景的命令需要两种已实现模型、非零训练步检查点和采样配置锁；最终评测只接受锁定的检查点。

`evaluate` 每步更新观测历史，每次执行动作块前四步。主阈值87%越过后继续执行，直到95%严格越过、环境终止或300步。结果包含原始完整计划、逐步实际动作与覆盖率；`recompute` 从轨迹重算指标，不相信缓存汇总。

`render_report` 对每个无序输入配对检查场景和协议兼容性；输出数据文件和人读报告。`checkpoint_score` 为 `(success_count_87,mean_max_coverage,-step)`，由你在训练的开发评测阶段调用。`lock-selection` 锁定你已选定的检查点；它不替你运行训练、遍历所有训练检查点或判定模型是否达到期望能力。

## 交互边界

页面状态只存在于独立会话。控制接收与推理并行，待处理拖动等命令被合并为最新值；较新的控制会使正在生成的动作块失效。输出队列只保留最新帧。断线取消异步任务后，会等待原生物理/推理线程结束再关闭环境。

默认只监听本地回环地址。浏览器 WebSocket（Web Socket，网页双向通信连接）要求与页面同源；无来源头的本地程序客户端可连接。交互场景台账只用于排除调试种子/姿态，不写正式结果。程序化自建 `PolicyRegistry` 必须显式传入 `debug_ledger` 才持久化；命令行服务已默认开启。
