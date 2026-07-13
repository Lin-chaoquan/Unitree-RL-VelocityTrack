# 强化学习实验调参与诊断指南

本文面向 Isaac Lab 中基于 PPO 的连续控制任务，总结从物理系统、MDP、奖励函数到 PPO 优化器的系统调参方法。文中的双摆示例来自当前仓库的 `cartpole_learn` 实验，但方法同样适用于机器人运动控制、机械臂操作和其他欠驱动系统。

## 1. 调参的基本框架

调参不是看到曲线异常后随意修改一个数字，而是沿着完整因果链定位问题：

```text
物理状态
  → 观测
  → 策略分布
  → 动作裁剪与缩放
  → 执行器和动力学响应
  → 奖励
  → 回报与优势估计
  → PPO 更新
  → 新策略
```

建议始终按以下顺序诊断：

1. **物理可行性**：给定足够且方向正确的动作，系统能否完成任务？
2. **MDP 正确性**：策略是否获得了完成任务所需且定义正确的状态信息？
3. **奖励正确性**：目标行为是否确实比其他行为获得更高回报？
4. **时间尺度匹配**：rollout、折扣和控制周期是否覆盖关键物理过程？
5. **PPO 优化效率**：策略是否有足够探索，更新是否稳定？

如果物理定义、角度定义或奖励目标有问题，增加训练轮数通常只会让策略更稳定地收敛到错误行为。

## 2. 常见现象与优先检查项

| 现象 | 优先检查 | 常见调整方向 |
| --- | --- | --- |
| 系统几乎不动 | action scale、策略动作、clip、阻尼、effort limit | 检查实际 applied effort，再调 scale/std |
| 有动作但幅度不足 | actor mean/std、执行器阻尼、动作惩罚 | 增加有效控制能力或探索，避免只看名义上限 |
| 动作剧烈或高频抖动 | action scale、std、entropy、action-rate、控制频率 | 降低物理探索，提高动作平滑约束 |
| 能到目标但无法静止 | 目标附近速度奖励、绝对速度定义、捕获奖励 | 加入 gated stationary/capture reward |
| 在目标附近来回振荡 | action-rate 太弱、局部奖励太宽、学习率偏大 | 增强平滑、缩窄捕获区、低学习率微调 |
| 停在目标附近但不进入中心 | 负向门控惩罚过强 | 降低门控惩罚，增加正向捕获奖励 |
| 只向一个方向推 | 对称性破坏、探索塌缩、局部最优 | 检查 std，重新训练，多 seed 验证 |
| 回报上升但视觉表现变差 | 奖励漏洞、总回报掩盖子项 | 分解奖励，增加独立成功指标 |
| 策略选择不动或“躺平” | alive 过大、动作/速度惩罚过强 | 降低无条件奖励和正则项 |
| 平衡能学会但 swing-up 失败 | 长期信用不足、缺少能量进度、reset 太难 | 增长实际 rollout 时间，加 curriculum |
| checkpoint 后期退化 | 学习率过高、std 异常、KL 过大 | 回选 checkpoint，低学习率 warm-start |
| 多个 seed 差异很大 | batch 小、奖励稀疏、初始化敏感 | 增加样本，改善 shaping，至少评估 3 个 seed |
| episode 很短 | termination 太严、动作过猛 | 检查终止原因与 reset 分布 |
| episode 很长但没有进展 | 没有失败定义、alive 占优 | 增加任务进度和独立成功指标 |

## 3. 从策略动作到真实物理作用

连续动作任务通常经历以下映射：

```text
a_policy ~ Normal(mean, std)
a_exec = clip(a_policy, -clip_actions, clip_actions)
u_command = action_scale × a_exec
```

### 3.1 Action scale

若 `action_scale=200 N`，动作 `0.25` 对应约 `50 N`。Action scale 同时决定：

- 最大控制能力；
- 网络输出到物理量的分辨率；
- 相同 policy std 对应的物理探索幅度；
- 动作饱和后的物理破坏程度。

scale 太小会导致系统没有足够能量；scale 太大会让很小的网络输出对应巨大作用，降低目标附近的精细控制能力。

### 3.2 策略 std 与物理探索

初始物理探索尺度近似为：

```text
physical_action_std ≈ policy_std × action_scale
```

例如：

```text
policy_std = 0.5
action_scale = 200 N
physical_std ≈ 100 N
```

如果 scale 从 100 增加到 400，同时希望保持原来的物理探索，应近似调整：

```text
std_new = std_old × scale_old / scale_new
```

原来 `100×0.5=50 N`，改为 scale 400 后，应使用约 `std=0.125` 才保持 50 N 的探索尺度。

### 3.3 执行器阻尼与有效作用

对于带速度阻尼的执行器，可近似理解为：

```text
u_effective ≈ u_command - damping × velocity
```

例如：

```text
u_command = 20 N
damping = 10 N·s/m
velocity = 5 m/s
```

阻尼项约为 50 N，因此系统未必沿命令方向继续加速。看到“推力不足”时，必须区分：

- 策略输出太小；
- scale 太小；
- 动作被 clip；
- 阻尼抵消；
- effort limit 限制；
- 系统质量或惯量过大。

建议记录：

```text
raw action
clipped action
applied effort
action saturation ratio
position / velocity / acceleration
```

### 3.4 Effort limit

`effort_limit_sim` 是执行器允许的上限，不代表策略一定能使用到该上限。例如 effort limit 为 400 N，而 action scale 为 100 N，策略的 feed-forward 动作通常最多仍只有 100 N。

## 4. 控制周期与物理时间尺度

### 4.1 仿真与控制周期

```text
physics_dt = sim.dt
control_dt = sim.dt × decimation
control_frequency = 1 / control_dt
```

例如：

```text
sim.dt = 0.01 s
decimation = 2
control_dt = 0.02 s
control_frequency = 50 Hz
```

控制频率太高时，固定步数 rollout 覆盖的真实时间短，策略也更容易产生高频动作。控制频率太低时，精细稳定能力和快速响应能力可能不足。

### 4.2 Rollout 的真实持续时间

```text
rollout_time = num_steps_per_env × control_dt
```

若 `num_steps_per_env=128`、`control_dt=0.02 s`：

```text
rollout_time = 2.56 s
```

rollout 应尽量覆盖一个关键物理过程，例如一次完整摆动、一次落脚周期或一次抓取接触过程。

### 4.3 改变 decimation 时的联动

改变 decimation 会同时改变：

- 策略决策频率；
- rollout 的实际时间；
- gamma/lambda 对应的物理信用范围；
- 单步 action difference；
- episode 的控制步数；
- 动作在物理系统中的保持时间。

因此 decimation 不能作为孤立参数调整。

## 5. PPO 参数与算法原理

PPO 的核心裁剪目标可简化为：

\[
L^{clip}=\mathbb{E}\left[\min\left(r_t A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right]
\]

其中 `r_t` 是新旧策略对同一动作的概率比，`A_t` 是优势函数，`ε` 对应 `clip_param`。PPO 不直接理解物理目标，它只根据优势判断某个动作是否比旧策略预期更好。

### 5.1 Learning rate、epochs、clip 与 KL

这些参数共同决定一次 PPO 更新能把策略推多远：

- learning rate 增大：每个梯度步更大；
- learning epochs 增大：同一批数据被重复优化更多次；
- clip_param 增大：允许更大的策略概率变化；
- desired KL 较小：adaptive schedule 更容易降低学习率。

如果 KL 经常超过目标、checkpoint 忽好忽坏、旧能力突然丢失，应优先降低 learning rate 或 epochs，而不是盲目增大 clip。

如果 KL 长期远低于目标、surrogate loss 很小、策略缓慢不动，可以检查奖励梯度后再考虑提高 learning rate 或 epochs。

常用经验范围：

```text
从零训练：learning_rate ≈ 3e-4 ～ 1e-3
已有策略微调：learning_rate ≈ 5e-5 ～ 2e-4
clip_param：通常先保持 0.2
```

### 5.2 Gamma 的物理意义

回报定义为：

\[
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
\]

Gamma 必须结合 control dt 转换为物理时间。折扣半衰期为：

\[
T_{1/2}=\Delta t_{control}\frac{\ln(0.5)}{\ln(\gamma)}
\]

例如在 50 Hz 控制下，`control_dt=0.02 s`、`gamma=0.99`，半衰期约为 1.38 秒。如果任务需要数秒后才能体现动作收益，可考虑 `gamma=0.995`，但这会增加回报方差和 value function 的学习难度。

若改变 control dt，又希望保持相同的物理折扣时间，可使用：

\[
\gamma_{new}=\gamma_{old}^{\Delta t_{new}/\Delta t_{old}}
\]

### 5.3 Lambda 与 GAE

GAE 近似为：

\[
\hat A_t=\delta_t+(\gamma\lambda)\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots
\]

`gamma×lambda` 决定优势信息传播多远：

- lambda 小：方差低，但更偏短期；
- lambda 大：长期信用更强，但对 critic 误差更敏感。

局部平衡通常可从 `gamma=0.99, lambda=0.95` 开始。多秒 swing-up 可尝试 `gamma=0.995, lambda=0.97～0.98`，前提是奖励和 critic 足够稳定。

### 5.4 Std 与 entropy

策略 std 决定探索强度，entropy reward 阻止 std 过快收缩。

Std 塌缩通常表现为：

```text
Policy/mean_std 持续降到 1e-3 以下
回报进入平台
策略动作接近固定
行为停在局部最优
```

处理方式包括：

- 使用 log-std 参数化；
- 合理增加 entropy coefficient；
- warm-start 时重设 std；
- 检查奖励是否允许“低动作躺平”；
- 改善探索阶段的 reset 和 shaping。

Entropy 太大则会让训练阶段长期随机，妨碍精细控制。常见分阶段设置：

```text
上摆/探索阶段：entropy_coef ≈ 0.005 ～ 0.01
稳定微调阶段：entropy_coef ≈ 0.001 ～ 0.003
```

推理通常使用 actor mean，因此播放时的持续振荡不一定由采样噪声直接造成，但训练中的高 entropy 会影响最终 mean policy。

### 5.5 Batch 与 mini-batch

```text
batch_size = num_envs × num_steps_per_env
mini_batch_size = batch_size / num_mini_batches
```

batch 太小会使梯度噪声和 seed 敏感性增大；batch 很大则会降低更新频率，并提高每次 value 拟合的规模。改变环境数或 rollout 时，应同步检查 mini-batch 大小，不要只保留原 mini-batch 数量。

## 6. 奖励函数与物理过程

Isaac Lab Manager-based reward 的基本计算为：

```text
step_reward = raw_value × weight × control_dt
```

因此 reward weight 可理解为每秒奖励率的比例。调权重前，应将典型状态代入计算，而不是只比较配置数字。

### 6.1 L1 与 L2

```text
L1 = |error|
L2 = error²
```

L1 对小误差仍有稳定梯度，对离群大误差不太敏感；L2 对大误差惩罚强，但可能让单一奖励项支配总回报，小误差附近的梯度也会减弱。

### 6.2 Gaussian 奖励中的 std

```text
reward = exp(-error² / std²)
```

| 误差 | 奖励值 |
| --- | --- |
| 0 | 1.000 |
| 0.5×std | 0.779 |
| 1×std | 0.368 |
| 2×std | 0.018 |

std 表示目标容差。宽 Gaussian 适合提供接近目标的引导，窄 Gaussian 适合最终精确捕获。复杂任务通常需要“宽进度奖励 + 窄捕获奖励”。

### 6.3 欠驱动系统的阶段冲突

只控制小车的双摆是欠驱动系统。典型过程是：

```text
小车加速
  → 杆获得动能
  → 适时换向
  → 动能转化为势能
  → 接近竖直
  → 制动
  → 局部稳定
```

这包含两个需求相反的阶段：

```text
能量注入阶段：允许较大速度和动作
捕获稳定阶段：角度和速度都应接近目标
```

全程惩罚速度会阻止上摆；完全不惩罚速度会导致高速穿过目标。因此速度惩罚通常应由目标接近程度门控。

### 6.4 门控速度惩罚

```text
near_target = exp(-(position_error) / std²)
velocity_penalty = near_target × velocity_error
```

这样远离目标时允许高速，接近目标后要求减速。但纯负向门控存在奖励漏洞：策略可以稍微离开目标区，使 gate 下降，从而减少速度惩罚。

例如 gate 从 1 降到 0.2、速度误差为 16、weight 为 -0.1：

```text
目标中心惩罚 = -1.6
稍微偏离惩罚 = -0.32
```

如果偏离造成的角度损失不到 1.28，策略会更愿意偏离。因此推荐组合：

```text
小幅 gated velocity penalty
+ 正向 stationary capture reward
```

正向捕获可写成：

\[
r_{capture}=\exp\left(-\frac{e_{pos}}{\sigma_p^2}-\frac{e_{vel}}{\sigma_v^2}\right)
\]

### 6.5 Action magnitude 与 action-rate

```text
action L2：不希望使用大动作
action-rate L2：允许大动作，但不希望快速抖动
```

Swing-up 通常需要大动作，因此 action L2 不宜过强；目标附近的抖动更适合通过 action-rate 和局部速度目标处理。

普通 action-rate 为：

```text
(a_t - a_{t-1})²
```

它不是严格的动作时间导数。改变 control dt 后，相同物理动作变化率对应的单步差值会改变。若希望获得更明确的物理意义，可使用：

```text
((a_t - a_{t-1}) / control_dt)²
```

## 7. 串联机构中的相对量和绝对量

双摆中：

```text
q1：第一杆角度
q2：第二关节相对第一杆的角度
第二杆绝对角度 = q1 + q2
第二杆绝对角速度 = dq1 + dq2
```

如果目标是第二杆在世界坐标中竖直，应使用：

```text
wrap_to_pi(q1 + q2 - target_absolute_angle)
(dq1 + dq2)²
```

不能简单使用 `q2²` 或 `dq2²`。例如 `dq1=2, dq2=-2` 时，第二杆绝对角速度为零；若惩罚 `dq1²+dq2²`，会错误地对实际静止的第二杆产生 8 的惩罚。

对依赖列顺序的奖励函数，必须明确设置 `SceneEntityCfg(..., preserve_order=True)`，避免关节按 USD 内部顺序返回后产生静默错误。

## 8. Reset、curriculum 与状态覆盖

Reset 分布定义了策略实际看到的任务难度：

```text
目标附近 reset：学习局部稳定器
大范围 reset：学习恢复和 swing-up
```

如果所有环境都从远处 reset，目标附近样本比例很低，策略可能学会上摆却不会精细稳定。如果都从目标附近 reset，则不会学习恢复能力。

推荐两种方式：

### 8.1 Curriculum

```text
阶段 1：目标附近 ±0.1 rad
阶段 2：扩大到 ±0.25π
阶段 3：扩大到 ±0.5π
阶段 4：完整任务
```

### 8.2 混合 reset

```text
50% 环境从目标附近开始
50% 环境从完整任务分布开始
```

混合 reset 有助于避免学习稳定后遗忘 swing-up，或学习 swing-up 后缺少目标附近精细样本。

## 9. 多参数耦合与联动调参

### 9.1 时间尺度耦合组

```text
sim.dt ↔ decimation ↔ rollout steps ↔ gamma/lambda ↔ action-rate ↔ episode steps
```

改变 decimation 后，应重新计算 rollout 秒数、折扣半衰期、episode 控制步数和 action-rate 的物理尺度。

### 9.2 动作探索耦合组

```text
action scale ↔ policy std ↔ entropy ↔ clip ↔ damping ↔ effort limit
```

调节目标不是让单个数字看起来合理，而是让实际物理探索、饱和率和有效作用处于合理范围。

### 9.3 目标捕获耦合组

```text
angle L2 ↔ Gaussian std ↔ velocity gate ↔ capture reward ↔ action-rate
```

角度奖励决定靠近目标的动力；velocity gate 决定何时制动；capture reward 防止回避目标；action-rate 决定控制命令是否平滑。

### 9.4 Reset 与奖励覆盖耦合组

```text
reset range ↔ curriculum ↔ episode length ↔ global progress ↔ local capture
```

奖励只在某状态区域有效时，reset 和策略轨迹必须保证该区域有足够样本。

### 9.5 优化稳定性耦合组

```text
reward scale ↔ critic/value loss ↔ learning rate ↔ epochs ↔ desired KL
```

即使 advantage 被标准化，整体奖励放大仍会放大 value target，影响 critic 稳定性，并间接污染优势估计。

## 10. 多参数实验方法

### 10.1 先定义希望保持的不变量

例如：

```text
物理探索力保持 50 N
rollout 实际时间保持 3 秒
折扣半衰期保持 2 秒
动作变化率惩罚保持一致
```

调整 scale、dt 或 decimation 时，通过公式同步换算相关参数，而不是独立修改。

### 10.2 按参数块调节

建议将参数分块：

- 物理控制块：scale、damping、control frequency；
- 探索块：std、entropy、clip；
- 奖励块：progress、capture、velocity、action-rate；
- 信用块：rollout、gamma、lambda；
- 优化块：learning rate、epochs、mini-batches、KL。

一次主要调整一个参数块，保持其他块不变。

### 10.3 使用 2×2 交互实验

怀疑两个参数存在耦合时，不要只做单变量实验。例如：

| 实验 | action-rate | stationary |
| --- | --- | --- |
| A | 小 | 小 |
| B | 大 | 小 |
| C | 小 | 大 |
| D | 大 | 大 |

如果 B、C 都无明显改善，而 D 有效，说明两个因素必须共同作用。

### 10.4 奖励预算表

为若干代表状态计算每项奖励：

| 状态 | 角度误差 | 角速度 | 希望的奖励结构 |
| --- | --- | --- | --- |
| 远离目标 | 大 | 可大 | progress 主导，速度约束弱 |
| 接近目标 | 中 | 中 | 角度、制动和 capture 同时生效 |
| 成功捕获 | 小 | 小 | stationary capture 达到最大 |

如果某个正则项在典型状态下比所有任务奖励之和还大，它很可能会改变任务本身。

### 10.5 短实验筛选

```text
100～200 iterations：排除明显错误组合
约 500 iterations：比较趋势和交互
完整训练：只运行最优少数组合
多 seed：最终确认稳定性
```

不要为每一个猜测直接运行上千 iterations。

### 10.6 控制实验变量

比较参数时应尽量保持：

- 相同 seed；
- 相同初始 policy checkpoint；
- 新 optimizer；
- 相同训练步数；
- 相同评估初始状态；
- 相同 deterministic evaluation 流程。

至少使用 3 个 seed，避免把随机初始化差异误认为参数效果。

## 11. Resume 与 warm-start

完整 `--resume` 通常会恢复：

- actor/critic；
- optimizer 和动量；
- policy std；
- observation normalizer；
- iteration。

它适用于环境、奖励、观测和动作完全相同，仅训练被中断的情况。

以下情况更适合 warm-start：

- 修改奖励；
- 修改目标；
- 修改 action scale；
- 希望使用新 optimizer；
- 希望重设 std；
- 进入低学习率稳定微调阶段。

Warm-start 通常只加载 actor/critic，按需保留 normalizer，重建 optimizer，并使用新实验目录。如果观测或网络维度改变，则通常无法直接加载旧权重。

## 12. 双摆实验的具体调参案例

当前任务相关文件：

- [环境配置](source/isaaclab_tasks/isaaclab_tasks/manager_based/cartpole_learn/cartpole_learn_env_cfg.py)
- [奖励函数](source/isaaclab_tasks/isaaclab_tasks/manager_based/cartpole_learn/mdp/rewards.py)
- [PPO 配置](source/isaaclab_tasks/isaaclab_tasks/manager_based/cartpole_learn/agents/rsl_rl_ppo_cfg.py)

### 12.1 有上摆趋势但推力不足

诊断链：

```text
检查 actor action 和 std
  → 检查 clip 后动作
  → 检查 applied force
  → 检查 damping 后有效作用
  → 检查动作惩罚是否压制探索
```

如果 scale 为 400，但实际 RMS 力只有 20 N，问题主要不是 scale 上限，而是策略只输出约 0.05 的动作。继续增大 scale 可能降低精细控制能力，而不能解决探索或奖励问题。

### 12.2 能竖直但无法静止

若所有中后期 checkpoint 都有较高角速度，说明不是简单的 checkpoint 选择问题，而是静止状态没有足够奖励优势。

建议：

```text
使用 dq1 + dq2 表示第二杆绝对角速度
两杆共同接近目标时才开启 stationary penalty
增加正向 stationary capture
适度增加 action-rate
降低微调 learning rate 和 entropy
使用目标附近与远处混合 reset
```

### 12.3 加入 stationary 后不愿进入目标

这通常是负向门控惩罚过强。可联动调整：

```text
stationary weight：-0.1 → -0.03/-0.05
capture weight：增加
gate std：适度增大
```

不要简单大幅增加角度 L2，否则可能重新产生暴力控制。

### 12.4 进入目标后仍有高频小抖动

若角度、速度已经较小，但动作每个控制周期快速正负切换，应优先调节控制序列而不是继续提高速度惩罚：

```text
action-rate：适度增强
entropy：探索阶段值降到微调阶段值
learning rate：降低
必要时降低 action scale 或引入动作低通/延迟模型
```

### 12.5 稳定后失去 swing-up

说明局部稳定目标压制了全局能量注入。应使用：

- 角度门控的局部速度/action-rate；
- 混合 reset；
- 全局 progress/energy reward；
- 局部 capture reward；
- 分阶段但避免灾难性遗忘的 curriculum。

## 13. 模型选择与独立指标

不要仅按 `Train/mean_reward` 选择 checkpoint。至少记录：

```text
任务成功率
首次到达目标时间
稳定持续时间
角度 RMS
绝对角速度 RMS
cart drift
action RMS
action-rate RMS
动作饱和率
policy std
终止原因
```

例如模型 A 的总回报更高，但稳定成功率只有 40%；模型 B 总回报略低，稳定成功率为 85%。如果真实目标是静止平衡，应选择 B，并据此修正奖励与真实指标的偏差。

适合继续训练的信号：

- 真实成功率持续上升；
- 角度和速度指标同步改善；
- std 没有异常塌缩；
- 最佳 checkpoint 接近训练末尾。

必须修改奖励的信号：

- 所有 checkpoint 都存在同一种行为缺陷；
- 回报增加但真实目标不改善；
- 策略稳定利用奖励漏洞；
- 真实目标量根本没有进入奖励。

## 14. 最终检查清单

开始长时间训练前，逐项确认：

- [ ] 固定动作能够驱动物理系统完成关键运动；
- [ ] 观测包含完成任务所需的状态；
- [ ] 相对角度、绝对角度和速度定义正确；
- [ ] 依赖关节列顺序的配置使用 `preserve_order=True`；
- [ ] action scale、std 和 clip 对应合理的物理探索；
- [ ] rollout 覆盖关键物理过程；
- [ ] gamma/lambda 已换算为合理的物理信用时间；
- [ ] 奖励在远处、接近目标和成功状态下均做过数值预算；
- [ ] 上摆与稳定阶段的奖励不直接冲突；
- [ ] 门控负奖励没有明显的目标回避漏洞；
- [ ] reset 分布覆盖全局任务和局部稳定状态；
- [ ] 日志记录真实物理指标，而不只有总回报；
- [ ] 修改任务后使用新实验目录和适当的 warm-start；
- [ ] 关键结论至少由多个 seed 验证。

## 15. 核心原则

调参最终是在匹配三个尺度：

1. **物理尺度**：N、N·m、m/s、rad/s、惯量和阻尼；
2. **时间尺度**：physics dt、control dt、rollout、gamma/lambda 和 episode；
3. **优化尺度**：reward、advantage、value target、learning rate、KL 和策略 std。

高质量调参不是找到一个“神奇数字”，而是让这些尺度在同一个任务目标下相互一致，并通过可复现的对照实验验证因果关系。
