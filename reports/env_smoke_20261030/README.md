# 环境与 G1 转换冒烟检查

环境使用 Python 3.10.12、PyTorch 2.7.1+cu128；B200 上执行了小张量的 CUDA 前向和反向计算。Kimodo、MotionCorrection、MuJoCo 与 PyArrow 均能导入，`uv pip check` 通过。

从已下载的原始 G1 CSV 中取 240 帧：`g1/csv/230425/jump_ff_360_R_001__A351_M.csv`，按现有 G1 数据处理代码将厘米、欧拉角和关节角转换为 qpos。`kimodo_convert` 将 120 Hz 下采样至 30 Hz，输出 60 帧；NPZ 中七个字段和导出的 `[60,36]` qpos 均为有限值。与输入每四帧比较，root 位置最大差约 `2.9e-8 m`、四元数旋转最大夹角约 `1.0e-5°`、关节角最大差约 `0.0125°`。这只是单条动作的环境冒烟检查，不代表全量数据或人体动作质量已通过核验。

已保留另一条来自 `seed5000_0501` 的 50 Hz 转换失败样本。`input_50hz_qpos.csv` 的 100 帧均为有限值，`kimodo_convert --source-fps 50` 输出的 `FAILED_50hz_motion_30hz.npz` 和 `FAILED_50hz_qpos_30hz.csv` 包含 NaN。最小检查发现当前 `kimodo/exports/motion_io.py` 的 `_quaternion_slerp` 对完全相同的两个单位四元数返回零四元数；后续矩阵转换会产生非有限值。120→30 Hz 的整数倍抽帧不经过该插值分支。50→30 Hz 转换必须修复并通过回归后才能用于训练；本次没有改动其他项目的 Kimodo 源码。
