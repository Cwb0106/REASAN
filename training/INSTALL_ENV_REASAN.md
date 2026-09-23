# 修复 env_reasan 安装

适用于已有 Python 3.10 和 Isaac Sim 4.5.0 的 `env_reasan` 环境。
使用项目中的 IsaacLab 和自定义 RSL-RL 源码。

`nvcc` 显示的本机 CUDA Toolkit 12.1 不需要替换。官方 PyTorch
`cu128` 包会安装配套 CUDA 运行库；本机已检查的 NVIDIA 驱动
580.178.04 支持它。编译自定义 CUDA 扩展时需要另外检查 Toolkit。

## 安装步骤

在 REASAN 仓库的 `training` 目录运行。遇到命令失败时，先解决该条
命令的错误，再继续后面的步骤。

```bash
conda activate env_reasan
cd /home/galbot/project/REASAN/training

# build constraints 需要 pip 25.3 或更新版本。
python -m pip install 'pip>=25.3'
export PIP_BUILD_CONSTRAINT="$PWD/requirements-reasan-build-constraints.txt"

# 修复 flatdict 构建，并先移除 SB3 2.9 对 torch>=2.8 的要求。
python -m pip install --index-url https://pypi.org/simple \
  'setuptools==80.9.0' 'flatdict==4.0.1' \
  'stable-baselines3==2.7.0' 'gymnasium==1.2.0'

# 约束后续安装的关键依赖，避免再次被自动升级。
export PIP_CONSTRAINT="$PWD/requirements-reasan-constraints.txt"
python -m pip install \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128

# 直接安装项目所需组件。让 pip 同时解析依赖，并安装本地 RSL-RL。
python -m pip install --index-url https://pypi.org/simple \
  -e ./IsaacLab/source/isaaclab \
  -e ./IsaacLab/source/isaaclab_assets \
  -e ./IsaacLab/source/isaaclab_tasks \
  -e ./IsaacLab/source/isaaclab_rl \
  -e ./IsaacLab/source/isaaclab_mimic \
  -e ./rsl_rl -e ./go2_lidar -e ./ray_predictor

python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python -c 'from rsl_rl.runners import OnPolicyRunnerLoco; import rsl_rl; print(rsl_rl.__file__)'
```

检查结果应包含 `torch 2.7.0+cu128`、CUDA `12.8`、GPU 可用 `True`，
以及指向本仓库 `training/rsl_rl` 的路径。

## 首次启动 Isaac Sim

Isaac Sim 首次启动会要求接受 NVIDIA Omniverse 许可协议：
https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html

在交互终端中运行：

```bash
cd /home/galbot/project/REASAN/training/IsaacLab
python scripts/tutorials/00_sim/create_empty.py --headless
```

阅读并同意协议后，在 `Do you accept the EULA? (Yes/No):` 提示处输入
`Yes`。看到 `[INFO]: Setup complete...` 表示空场景已启动；按 Ctrl+C
结束。首次启动的许可提示在非交互脚本中可能显示 `EOFError`，需要先完成
许可确认。

## 避免再次覆盖版本

原命令 `./isaaclab.sh --install` 会安装全部可选框架，并安装 PyPI 的
`rsl-rl-lib==2.3.3`，覆盖项目自带的 2.3.1。重新安装时使用上面的直接
安装命令；如果使用 `./isaaclab.sh --install none`，仍需保留这两个约束
环境变量，并检查核心 `isaaclab` 是否安装成功。原脚本可能在一个组件失败
后继续执行后续安装。

约束环境变量仅对当前 shell 及其子进程有效。在新终端继续安装依赖时，
重新设置这两个绝对路径；执行 `unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT`
可取消当前终端中的约束。

末尾缺少 `.vscode/tools/setup_vscode.py` 的警告只影响编辑器配置。

## 本机验证记录（2026-09-20）

- IsaacLab 核心及其 assets、tasks、rl、mimic 组件均已安装。
- `rsl-rl-lib 2.3.1` 来自本仓库，两个自定义 runner 均可导入。
- `go2_lidar` 和 `ray_predictor` 以 editable 方式安装。
- `pip check` 输出 `No broken requirements found.`。
- PyTorch 2.7.0+cu128 在 RTX 4090 上完成矩阵乘法，与 CPU 结果一致。
- flatdict、wandb、SB3、Pinocchio、Pink、Transformers 导入检查通过。
- Isaac Sim 首次启动停在 EULA 确认，尚未完成仿真启动验证。
