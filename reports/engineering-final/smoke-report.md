# Push-T 评测报告

证据类别：formal = 正式评测；real_smoke = 真实短闭环工程检查；mock = 模拟接口检查。

| 策略 | 证据类别 | 场景数 | >87% 成功 | >95% 成功 | 平均最大覆盖率 | 规划中位耗时（毫秒） |
|---|---|---:|---:|---:|---:|---:|
| act | real_smoke | 1 | 0 | 0 | 0.1457 | 7.355542002187576 |

来源：`/Users/marscolonizer/Documents/ChatGPT/Diffusion/pusht-diffusion/reports/engineering-final/act-smoke.json`；文件校验：`938cd3479ff31132a2641bfab284daa1175810d3b9b769ac196b21adc51d7225`。
87% 成功率的 Wilson 95% 置信区间：[0, 0.7934506856227626]；95% 阈值对应区间：[0, 0.7934506856227626]。

## 配对结果


以上区间只反映本批场景的不确定性，不代表跨训练种子的稳定性。真实短闭环和模拟检查不得解释为策略正式效果。
