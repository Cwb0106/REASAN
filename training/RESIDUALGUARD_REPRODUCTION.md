# ResidualGuard / REASAN 实现与论文逐项核对

完成日期：2026-09-21。依据仓库内 [ResidualGuard 论文](../ICRA2027_ResidualGuard.pdf) 第 3–5 页方法、公式及第 5–7 页实验要求。

**结论：已完成 Go2 平台的方法实现，真实 Isaac Sim 中的强化学习、感知训练、断点恢复和导出推理链路均已跑通。没有训练到收敛，没有复现论文表格数值，也没有把普通 Go2 声称为 Go2-W。**

此前的路线文档保存在 [RESIDUALGUARD_REPRODUCTION_PLAN_20260920.md](RESIDUALGUARD_REPRODUCTION_PLAN_20260920.md)。本文描述实际交付。

## 1. 完成范围和差异

已实现独立 ResidualGuard 环境、三维残差、Ray-DCR、风险门控奖励和 actor 正则、响应编码器、recurrent PPO、独立辅助 optimizer、CNN/FiLM/ConvGRU 感知网络、数据采集/隔离、训练/播放/恢复/导出及消融配置。

必须明确的差异：

- **平台：**论文为 Go2-W，当前为 REASAN Go2、12 关节动作和原冻结步态策略。历史每帧为 45 维，不补零冒充论文 57 维；不声称实现 4 m/s。
- **参数：**论文未公开完整风险系数、奖励权重、候选运动代理、探索规则和 critic 等。本次全部显式配置，不把实现选择称为作者原始参数。
- **场景：**沿用 REASAN 程序化地形、动态障碍；未复刻完整 8/12/15 柱 Forest、三阶段 Corridor、480 episode 配对协议和实机实验。
- **感知：**跨帧位姿和时间条件已实现；现有仿真每帧同时刻射线采样，未验证真实逐点运动补偿、FAST-LIO2、ROS 2、TensorRT、SDK2。
- **碰撞裁判：**继承非足部接触/姿态判断，增加足部侧碰撞近似。地面和静态柱体共用 terrain mesh，尚不是论文评测所需的精确任意部位障碍接触裁判。

## 2. 文件改动

以下路径相对 `training/`。

| 文件 | 用途 |
| --- | --- |
| [config.py](residualguard/config.py)、[go2.json](residualguard/configs/go2.json) | 集中定义、校验并保存风险/控制/PPO参数 |
| [risk.py](residualguard/risk.py) | Ray-DCR 式(4)–(8)、TTC对照、门控 |
| [control.py](residualguard/control.py) | 残差处理、奖励、raw-mean正则 |
| [models.py](residualguard/models.py) | encoder、actor/critic、部署模块 |
| [runner.py](residualguard/runner.py) | 时序rollout、GAE、PPO、辅助更新、恢复/导出 |
| [go2_residualguard_env.py](go2_lidar/go2_lidar/tasks/go2_residualguard_env.py) | 动作前风险、步态调用、观测历史、监督目标、奖励/reset |
| [go2_residualguard_env_cfg.py](go2_lidar/go2_lidar/tasks/go2_residualguard_env_cfg.py) | Go2环境与感知来源配置 |
| [tasks/__init__.py](go2_lidar/go2_lidar/tasks/__init__.py) | 注册 `Unitree-Go2-ResidualGuard` |
| [go2_filter_env.py](go2_lidar/go2_lidar/tasks/go2_filter_env.py) | 支持配置化步态路径、地形行列数及跳过旧预测器；原任务默认行为保留 |
| [perception.py](residualguard/perception.py) | CNN/FiLM/ConvGRU、运动历史、式(12)损失 |
| [data.py](residualguard/data.py) | HDF5、因果窗口重建、episode隔离 |
| [train_residualguard.py](scripts/train_residualguard.py) | IsaacLab RL训练入口 |
| [play_residualguard.py](scripts/play_residualguard.py) | 导出策略播放、reset/命令断言、轨迹记录 |
| [collect_residualguard_clearance.py](scripts/collect_residualguard_clearance.py) | 感知采集 |
| [train_residualguard_clearance.py](scripts/train_residualguard_clearance.py) | 感知训练/验证/保存/导出 |
| [evaluate_residualguard_clearance.py](scripts/evaluate_residualguard_clearance.py) | 独立测试Critical MAE/DOR |
| [verify_residualguard.py](scripts/verify_residualguard.py)、[tests](tests/test_residualguard.py) | 真实仿真链路验证与核心测试 |
| [configs/](residualguard/configs/) | 完整模型、Res-TTC、去切向项、去response latent |
| `../.gitignore` | 忽略新训练产物，不忽略源码/JSON |

没有修改原 RSL-RL PPO/runner。原 runner 存在策略类选择硬编码，新建独立 runner 明确处理 context 缓存、时序批次和两个 optimizer。

接手时 loco/filter 导出权重已相对 Git 有修改；本次没有覆盖，验证脚本检查运行前后 SHA-256 一致。

## 3. 环境和控制时序

实测：Python3.10、PyTorch2.7.0+cu128、IsaacSim4.5.0.0、IsaacLab0.41.3、本地rsl-rl-lib2.3.1、Gymnasium1.2.0、h5py3.16.0、RTX4090。`pip check` 通过；没有重装环境。

| 项目 | 默认值 |
| --- | --- |
| 物理dt / decimation | 0.005s / 4 |
| residual/步态调用频率 | 50Hz，控制dt=0.02s |
| 并行环境 / episode | 256 / 9s |
| 地形 | REASAN粗糙地面、静态障碍、边界，10×10个9m×9m地块 |
| 动态障碍 | 最多3个/环境，0.5–1.5m/s；初始至少1个，保留少量无障碍样本 |
| 名义采样范围 | `[1.5,0.6,1.0]`，含向前/一般平面运动/静止及原上游返回指令 |
| 执行命令限制D | `diag(2.5,1.5,3.0)`，m/s、m/s、rad/s |
| 冻结步态 | `logs/rsl_rl/go2_lidar/loco_1/exported/policy.pt`，45维输入/12维输出，eval且无梯度 |

继承随机化：静/动摩擦0.2–1.25、恢复系数0–1、基座质量增量−1至+2kg、质心xy±0.05m/z±0.02m、电机强度0.9–1.1及外力/推扰。父环境创建时把PD随机化到 `35×[0.9,1.1]`、`0.5×[0.9,1.1]`；这是运行值，与资产cfg初始刚度25不同。

每步顺序：

1. 在 `s_t` 读取名义命令、命中点、物体速度、机体运动，构建观测/历史并计算 `R_nom`。
2. encoder估计速度/latent，actor采样无界动作，保存当时context、风险和分布。
3. 仅平滑残差、相加/裁剪，在同一 `s_t` 计算 `R_cand`，然后调用冻结步态。
4. 运行4个物理子步，在自动reset之前保存下一步响应、terminal critic输入及奖励。
5. 此后才生成下一步名义命令。终止环境清空残差、步态状态、历史和感知窗口；runner重置actor LSTM。

风险几何为基座中心、按yaw对齐的平面坐标。GT用单水平中心、3高度×180射线，选择每个方位的最近有效命中及对应mesh id。无命中不能由裁剪后的4m值伪造为障碍。明显roll/pitch时该平面坐标与完整机体坐标有差异，本实现沿用姿态终止条件限制其适用范围。

## 4. 网络结构

```text
5×45历史 -> flatten225 -> 128 ELU -> 64 ELU
                                      ├─ Linear3：归一化机体线速度估计
                                      └─ Linear8 -> L2 normalize：latent

180 rays -> 128 ELU -> Linear64 ───────────────┐
速度估计3 ───────────────────────────────────┼─ 79 -> 单层LSTM256
IMU6 + nominal3 + previous_filtered_delta3 ───┘          │
                                                拼接latent8
                                                       ↓
                                        264 -> 512 -> 256 -> 128 -> Linear3
                                               ELU

critic: 736 -> 512 ELU -> 256 ELU -> 128 ELU -> Linear1
辅助头: latent8 + current_response3 + executed3 -> 64 ELU -> 32 ELU -> Linear3
```

- 历史单帧：角速度×0.25(3)、投影重力(3)、相对默认关节位置(12)、关节速度×0.05(12)、前次执行命令/D(3)、前次步态输出(12)，共45维；全部因果可用。
- actor proprio12：IMU6、nominal/D、上次filtered residual/D。
- 速度监督 `[vx,vy,vz]/[2.5,1.5,1.0]`；平面响应 `[vx,vy,ωz]` 和执行命令除以D。
- critic：proprio12、真值线速度3、yaw rate1、GT range180、障碍xy速度360、valid180，共736；actor不读取这些特权量。
- encoder/辅助头只由独立监督优化，PPO参数排除它们。
- 总参数1,259,541，其中encoder与辅助头41,038；感知网络另有307,249参数。

论文指定的5帧、128–64 encoder、3D速度、8D L2 latent、64D ray feature、256 LSTM、512–256–128 actor均保留。ray encoder内层、critic和辅助头完整配置未公开，采用上面显式结构。

## 5. 残差、风险和奖励参数

### 5.1 指令接口

```text
raw_delta = S_delta * action
filtered_delta = (1-beta)*previous_filtered_delta + beta*raw_delta
before_clip = nominal + filtered_delta
executed = clip(before_clip, -limits, limits)
effective_delta = executed - nominal
```

`S_delta=[1.0,0.6,1.0]`，`beta=0.4`。nominal不经EMA；action不经tanh、不预裁剪；S_delta只是单位换算。

### 5.2 Ray-DCR

实现实测速度扩张椭圆、仿射参考圆、径/切向速度、切向允许量、负裕量饱和、循环max pooling、指数权重聚合。显式 `−ωJp` 只进径向；无效ray在池化前清零；全零权重返回0。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| a0 / b0 | 0.52 / 0.38m | 基础风险椭圆；保守实现选择，非作者已知风险参数 |
| tau_x / tau_y | 0.15 / 0.15s | 实测速度扩张 |
| margin_max | 0.30m | 半轴额外裕量上限 |
| r0 | 0.38m | 参考圆半径 |
| alpha | 1.5/s | 间隙项 |
| tangent_gain | 1.0 | 切向增益 |
| allowance_max | 3.0m/s | 切向允许上限 |
| velocity_epsilon / range_epsilon | 0.05m/s / 1e−6m | 数值保护 |
| violation_scale / saturation | 1.0m/s / 1.0 | 负裕量尺度和饱和 |
| concentration | 4.0 | `expm1(eta*pooled_violation)` |
| pool_radius | 2 | 循环窗口宽度5 |
| proxy_gain | 0.5 | `measured+0.5*(command-measured)` |
| TTC horizon | 1.5s | 独立Res-TTC对照 |

公式按论文；未公开数值和短时运动代理为本次显式假设。

### 5.3 奖励和actor正则

令 `g=(1-R_nom)^2`、`e=(executed-nominal)/D`：

```text
reward = dt * [
   4*g*exp(-sum(e²)/0.25) + nominal_direction_cosine - g*sum(e²)
 - 0.05*sum(((raw_delta-previous_raw_delta)/D)²)
 - 5*sum(((before_clip-executed)/D)²) - 8*R_cand
] - 20*hard_failure
```

名义平移速度范数≤0.1m/s时方向项为0。tracking约束执行命令相对nominal的保真。terminal罚项不乘dt；训练时间上限截断不是hard failure。论文评测中的超时失败需独立定义，不能把训练timeout比例当成功率。

正则严格作用于raw分布均值：

```text
L_reg = mean_valid_nonterminal(
  stop_gradient(g) * mean_j(sqrt(1+(S_delta[j]*mu[j]/epsilon[j])²)-1)
)
epsilon=[0.2,0.15,0.3]，loss weight=0.02
```

正则只进优化目标，不进reward/return，门控使用R_nom。奖励子项的完整形式及权重未在论文公布，这里明确列出实现选择。

## 6. RL配置和更新流程

| 项目 | 默认值 |
| --- | --- |
| seed / iterations | 42 / 20,000，本次仅跑有限验证 |
| rollout | 每环境24步 |
| PPO epochs / mini-batches | 5 / 4 |
| optimizer / 起始LR | Adam / 3e−4 |
| LR schedule | Gaussian KL自适应，目标0.01，范围1e−5至1e−3 |
| clip | 0.2 |
| gamma / GAE lambda | 0.99 / 0.95 |
| value loss | coefficient1.0，clipped |
| entropy coefficient | 0.005 |
| max gradient norm | 1.0 |
| 初始std | 每维0.35 |
| 风险条件探索 | `exp(clamp(log_std,-5,1))*(0.5+R_nom)`；推理用均值 |
| encoder optimizer / LR | 独立Adam / 1e−3 |
| encoder epochs / batch | 2 / 512 |
| velocity / response loss | Smooth-L1，各权重1 |
| 保存周期 | 每100轮，以及正常结束时 |

默认batch为256×24=6144 transitions。按环境切minibatch，保持完整时间轴；序列内部episode起点重置LSTM，不打乱时间、不混入padding。

rollout记录context、action、mean/std、log probability、nominal risk、辅助目标和valid。每轮先PPO再独立训练encoder，整轮PPO使用记录的context，不重新编码旧样本。

timeout从reset前terminal critic输入自举；GAE不跨reset。辅助下一步目标也在reset前保存，终止/截断样本从辅助loss和正则排除。

checkpoint含模型、两个optimizer、iteration、配置和随机状态。恢复会重新开始物理episode/LSTM，不是完整仿真状态逐位恢复；同时核对冻结步态和感知权重哈希，避免恢复时悄悄换控制器/感知来源。

## 7. 感知网络、数据和损失

```text
8×(30×180 grids / 4m)
 -> circular CNN24/48/64，三个stride-2 block
 -> FiLM(IMU6 + motion9)，输出64通道scale/bias
 -> ConvGRU64，每个八帧窗口零初始化，旧到新融合
 -> 上采样decoder48/24 -> elevation mean pooling
 -> circular Conv1d -> sigmoid -> 180 ranges / 4m
```

motion9：相对平移xyz、相对旋转rotvec xyz、观测年龄、外推时长、validity；相对最新帧。同步仿真外推时长为0。无运动条件消融清零前8维，保留IMU/validity。

HDF5保存grid、IMU、位置、wxyz四元数、time、extrapolation、valid、GT米制标签、episode、step和seed。按原始帧保存，按需重建因果窗口；episode开头padding不更新ConvGRU。标签为simulator label time，不混淆实机延迟后的policy-use time。

```text
L = L_metric + 0.20*L_over + 0.25*L_edge + 0.50*L_temporal
h_d=0.10m，overestimation margin=0.10m
critical：GT radial gap to ellipse(0.52m,0.38m) <= 0.20m
```

metric/edge/temporal使用单位阈值Smooth-L1。edge为循环相邻，temporal比较同episode相邻帧误差变化；over取全局/critical均值各一半，空critical回退全局，无temporal pair则为0；无效/负GT排除。

默认Adam、LR1e−3、batch32、50epochs、梯度裁剪1。完整episode划分20%验证集，不按滑窗打散。独立测试用另一个文件和不同seed；Critical MAE/DOR先每episode统计再等权聚合，无critical episode时输出null。

## 8. 运行命令

```bash
conda activate env_reasan
cd /home/galbot/project/REASAN/training
python -m pip check
```

首次Isaac Sim加载需要缓存或资产服务可访问。命令均在training目录执行。

### 有限验证

```bash
# 使用新目录；HDF5采集拒绝覆盖。
python scripts/verify_residualguard.py --output logs/residualguard/my_verification
python scripts/verify_residualguard.py --output logs/residualguard/my_verification --heldout-only
python -m unittest discover -s tests -p 'test_residualguard.py' -v
```

smoke使用4环境、4×4地形、16步rollout、PPO2epochs/2minibatches、encoder1epoch、0.6s episode，主动触发timeout；不是正式训练参数。

### GT训练及恢复

```bash
python scripts/train_residualguard.py \
  --config residualguard/configs/go2.json --num-envs 256 --iterations 20000 \
  --log-dir logs/residualguard/go2_gt

# checkpoint必须实际存在，iterations为额外训练轮数。
python scripts/train_residualguard.py \
  --config logs/residualguard/go2_gt/config.json \
  --resume logs/residualguard/go2_gt/model_20000.pt \
  --num-envs 256 --iterations 2000 --log-dir logs/residualguard/go2_gt_resume
```

### SwanLab实验记录

ResidualGuard runner始终保留AFS中的`metrics.jsonl`、checkpoint和导出策略。传入
`--swanlab-project`时，额外把标量和完整配置记录到SwanLab；不传该参数则不导入
SwanLab，也不改变原训练行为。

```bash
# 只需安装一次。--isolated避免系统pip配置中的额外源影响安装。
python -m pip --isolated install swanlab \
  --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 在共享服务器上把凭证限制在当前仓库；API Key不会进入训练参数。
swanlab login --local

python scripts/train_residualguard.py \
  --config residualguard/configs/go2.json \
  --num-envs 256 --iterations 20000 \
  --log-dir logs/residualguard/go2_gt \
  --swanlab-project residualguard \
  --swanlab-experiment-name go2_gt_seed42
```

团队空间可追加`--swanlab-workspace <workspace_username>`，同一组实验可追加
`--swanlab-group <group>`。服务器无法访问SwanLab时使用
`--swanlab-mode offline`；日志保存在`<log-dir>/swanlab/run-*`，之后在能联网的
机器上执行：

```bash
swanlab sync logs/residualguard/go2_gt/swanlab/run-*
```

恢复同一个SwanLab实验时，除训练checkpoint外，还需传入原实验ID：

```bash
--swanlab-id <experiment_id> --swanlab-resume must
```

感知训练入口支持同样的SwanLab参数。SwanLab公有云能够记录标量和配置，但
`swanlab.save`上传checkpoint仅支持私有部署，因此模型继续以AFS文件为准。
SwanLab当前视频接口只接受GIF、不接受MP4；训练入口本身没有视频记录，所以本次
不增加转码和视频上传，播放生成的媒体继续保存在本地输出目录。

### 感知训练和预测射线闭环

```bash
python scripts/collect_residualguard_clearance.py \
  --num-envs 32 --steps 10000 --seed 42 \
  --output logs/residualguard/data/train.h5
python scripts/train_residualguard_clearance.py \
  --data logs/residualguard/data/train.h5 \
  --output logs/residualguard/perception --device cuda:0 --epochs 50 --batch-size 32
python scripts/train_residualguard.py \
  --config residualguard/configs/go2.json --num-envs 128 --iterations 20000 \
  --clearance-checkpoint logs/residualguard/perception/best.pt \
  --log-dir logs/residualguard/go2_predicted

# 独立测试，正式评测需要足够多critical样本。
python scripts/collect_residualguard_clearance.py \
  --num-envs 32 --steps 3000 --seed 43 \
  --output logs/residualguard/data/heldout.h5
python scripts/evaluate_residualguard_clearance.py \
  --data logs/residualguard/data/heldout.h5 \
  --checkpoint logs/residualguard/perception/best.pt \
  --output logs/residualguard/perception/heldout_metrics.json
```

采集可加 `--checkpoint <residual_checkpoint>`，覆盖策略访问状态。缺失预测器权重会报错，不回退GT。当前不放宽恢复的配置/感知检查；预测射线示例从头训练，不把更换感知条件混作普通resume。

### 播放和消融

```bash
python scripts/play_residualguard.py \
  --policy logs/residualguard/go2_predicted/exported/policy.pt \
  --config logs/residualguard/go2_predicted/config.json \
  --clearance-checkpoint logs/residualguard/perception/best.pt \
  --num-envs 4 --steps 1000 --output logs/residualguard/play
```

加 `--gui` 显示窗口，`--nominal` 执行零残差参考；有限步后保存rollout.npz和summary.json。使用真正导出模型，断言执行命令一致，任何Ray-DCR调用会失败。这不是论文benchmark，不输出伪造SR/TP。

训练尚未结束时也可直接加载任意训练checkpoint并在无界面模式录制MP4：

```bash
python scripts/play_residualguard.py \
  --checkpoint logs/residualguard/go2_gt_env512/model_2000.pt \
  --config logs/residualguard/go2_gt_env512/config.json \
  --output logs/residualguard/go2_gt_env512/eval/model_2000 \
  --num-envs 1 --steps 500 \
  --video --video-length 500
```

视频保存在`<output>/videos/*.mp4`，同目录还会保存`rollout.npz`和`summary.json`。
播放相机默认以`(3, 3, 2)`米偏移跟随第0个环境的机器人，可用
`--camera-eye X Y Z --camera-lookat X Y Z`调整构图。
checkpoint与config、locomotion policy及感知来源必须匹配；预测射线checkpoint还必须传入训练时的
`--clearance-checkpoint`。单卡上同时运行训练和第二个Isaac Sim会争用GPU/CPU，正式训练期间优先在
另一张GPU或单独算力任务中录制。

消融应独立重新训练：

| 消融 | 配置/开关 |
| --- | --- |
| Res-TTC | `residualguard/configs/go2_residual_ttc.json` |
| w/o tangential | `residualguard/configs/go2_no_tangent.json`，nominal/candidate都移除切向项 |
| w/o response latent | `residualguard/configs/go2_no_response_latent.json`，仅actor latent置零，保留速度头 |
| 感知w/o motion | `train_residualguard_clearance.py --no-motion-conditioning ...` |

## 9. 最终验证证据

目录：[verification_final](logs/residualguard/verification_final/)。命令、退出码、耗时见 [verification.json](logs/residualguard/verification_final/verification.json)。

| 验证 | 最终结果 |
| --- | --- |
| 核心测试 | 19/19通过：公式、切向/yaw、无效射线、循环边界、EMA、正则梯度、GAE、LSTM reset、优化器更新、恢复/导出、感知梯度和数据隔离 |
| 真实采集 | 4×32=128帧，有效HDF5 |
| 感知训练 | 1epoch，train/val各2batches；6 train episodes/2 validation episodes；反传、保存、导出通过 |
| 独立感知测试 | seed43另采64帧，4 episodes，其中2个含critical样本；测试入口通过，见[heldout_metrics.json](logs/residualguard/verification_final/heldout_metrics.json) |
| GT RL | 4×16×3=192 transitions，PPO/encoder loss有限 |
| 恢复 | iteration3恢复至4，再训练64 transitions，两个optimizer恢复 |
| 预测射线RL | 4×16×2=128 transitions，训练/导出通过 |
| 导出播放 | 4×40=160控制周期；命令一致/reset断言通过，Ray-DCR调用0次 |
| timeout | GT/预测训练及播放均触发，播放4次timeout、0硬终止 |
| 原权重 | loco/filter前后SHA-256一致 |
| 环境/静态 | pip check、Python编译、新代码Ruff检查通过 |

最终RL验证共 **6 iteration、384条真实仿真transition**。单元测试另有明确命名的runner contract fixture，不用它替代真实Go2证据。

短时播放平均IC约0.0214，感知验证MAE约1.44m；这是未收敛smoke诊断输出，不是避障性能，不能与论文比较。短测试0硬终止也不是论文SR。

独立感知小样本测试的Critical MAE为1.4194m、DOR为100%。预测器只更新了两个小batch，该结果验证的是独立采集与统计入口，说明这份smoke权重尚不能作为正式避障感知模型。

审查修复了足部误判：仅按水平力大小会把粗糙地面落脚当碰撞。最终要求水平力>5N且大于 `1.5*abs(竖直力)+1N`，识别以侧向法向力为主的接触；严格评测仍应换成精确裁判。

## 10. 论文逐项核对

“实现”表示机制已落地，“适配”表示平台/参数不同，“未验证”不表示已复现结果。

| 论文点 | 状态 | 核对结论 |
| --- | --- | --- |
| 式(1) residual相加/裁剪 | 实现 | 环境/导出一致，effective correction=executed−nominal |
| 式(2)5帧、128–64双头 | 适配 | 拓扑匹配，Go2每帧45维而非57维 |
| latent8归一化/LSTM后拼接 | 实现 | latent在LSTM后，速度估计在LSTM前 |
| ray64/LSTM256/MLP512-256-128 | 实现 | 指定结构匹配；未公开内层/critic采用本文配置 |
| 式(3)仅平滑residual | 实现 | nominal不平滑，无tanh/动作预裁剪 |
| encoder独立监督 | 实现 | 当前xyz速度和下一步平面响应Smooth-L1，独立Adam |
| PPO记录的context | 实现 | update不重新编码，测试参数归属及context不变 |
| 式(4)实测速度包络 | 实现 | 不随候选改变，实测速度固定于动作前 |
| 式(5)仿射几何/间隙 | 实现 | 独立标量参考检验式(4)–(8) |
| 式(6)候选/障碍运动、yaw径向 | 实现/假设 | 公式匹配，未公开代理使用gain0.5一阶混合 |
| 式(7)切向长度/允许量/cap | 实现 | 测试区分逼近与擦身 |
| 式(8)无效清零/循环池化/聚合 | 实现 | 池化前清零、无命中风险0、循环平移不改结果 |
| 同状态R_nom/R_cand | 实现 | 动作前几何缓存，物理前计算，奖励后换命令 |
| 式(9)candidate风险/dt/terminal | 实现/适配 | 归属/缩放匹配，未公开子项形式与权重显式列出 |
| 式(10)nominal gate/raw mean/valid | 实现 | detach风险、actor额外loss、不进return、排除终止 |
| 风险条件探索 | 实现/适配 | rollout/update同std规则，规则为本次选择 |
| 推理无DCR/response head | 实现 | 导出移除critic/辅助头，真实播放0次risk调用 |
| 式(11)8帧/30×180/180输出/4m | 实现 | base-centered surface ranges，不减footprint |
| CNN24/48/64、FiLM、GRU64 | 实现 | IMU6+motion9，相对最新帧，每窗口清零 |
| 点级补偿/真实延迟 | 未验证 | 同时刻仿真采样，外推为0，非完整实机前端 |
| 式(12)四项损失与数值 | 实现 | 0.20/0.25/0.50、0.10m、critical椭圆、同episode temporal |
| 感知独立训练、GT/预测区分 | 实现 | 两条RL链路实跑，预测器冻结 |
| Res-TTC/去切向/去latent/去motion | 实现入口 | 独立配置/开关，无论文消融性能声明 |
| 式(13)IC | 适配 | effective correction和Go2 D，固定dt均值等于时间加权均值 |
| SR/TP/Threat/配对场景 | 未复现实验 | 需独立nominal-continuation标签、精确裁判和统一协议 |
| Go2-W4m/s与实机 | 平台缺口 | 需轮足资产、执行器及高速步态，Go2权重不能替代 |

## 11. 正式实验的后续条件

用户要求的可执行方法与训练链路已完成。追求论文性能时，需长时训练与独立测试、Go2-W资产及高速步态，并在四种方法间共享机器人参数、感知、名义指令和配对场景。严格SR计入所有障碍/墙体接触、跌倒、越界、超时、控制中止；TP按论文起终点定义；Threat/nominal-safe由独立名义继续运行决定，不能由Ray-DCR自标注。

当前交付是含显式适配假设、可复现配置和真实训练证据的Go2方法实现；论文高速性能、统计结果和实机结果仍需对应实验支持。
