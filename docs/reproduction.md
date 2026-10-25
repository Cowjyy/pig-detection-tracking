# 复现说明

## 适用环境

完整控制器面向Linux、NVIDIA CUDA和独占测速GPU，正式实验使用RTX3090 24GB。在项目根目录执行命令。数据准备可在Windows运行；正式训练请使用服务器上的项目专用环境。

原始数据、模型权重、派生标签和完整日志不在Git仓库内，需获得数据授权后自行准备。`datasets/raw/inventory.json`记录本轮2051个原始文件的相对路径、字节数和SHA-256；mtime仅用于同机审计，跨平台核对使用`--ignore-mtime`。

```text
pig-objection detection/
  data/<编号>.jpg
  label/<同编号>.xml
PIG_MOT_TEST_DATA/
  MyVideo_191.mp4 ... MyVideo_197.mp4
  191/gt/gt.txt 和 labels.txt
  ...
  197/gt/gt.txt 和 labels.txt
```

两个原始目录保持只读用途。所有输出必须是新目录，禁止使用本轮正式运行目录作为新输出。

## 源码与依赖

`requirements.txt`只包含数据准备依赖。检测、跟踪环境分开建立：

| 环境 | 正式版本 |
|---|---|
| 两环境Python | 3.10 |
| 检测 | torch 2.1.0+cu121，torchvision 0.16.0，MMCV 2.1.0，MMEngine 0.10.7，MMDetection 3.3.0 |
| 跟踪 | torch 2.2.2+cu121，torchvision 0.17.2，BoxMOT 12.0.4 |
| 两环境NumPy | 1.26.4 |

实际依赖快照为`configs/environments/detection.lock.txt`和`tracking.lock.txt`，版本约束同目录可查。快照中的editable路径已改为项目相对路径，包版本未改。CUDA版PyTorch和MMCV须与对应环境匹配，不能只安装根目录requirements就开始训练。

先建立`.venvs/detection`与`.venvs/tracking`两个Python 3.10虚拟环境，依据上述约束安装相应PyTorch、MMCV、MMEngine等依赖，再执行：

```bash
python -m pip install -r requirements.txt
python -B scripts/bootstrap_sources.py
bash scripts/install_server_dependencies.sh
.venvs/detection/bin/python -B scripts/download_weights.py --models all --reid
```

源码获取固定提交，权重下载记录来源和SHA-256。OSNet采用作者MSMT17 combineall权重，三种ReID跟踪器共用同一文件；若官方源不可达，脚本中的备用源仍必须通过同一哈希校验。

## 检查仓库中的冻结记录

以下两项不读取数据本体，也不启动GPU实验：

```bash
python -B -m unittest discover -s tests -v
python -B scripts/verify_repository_release.py
```

正式检测划分在`datasets/splits/protocol_v2/detection/`，含660/195/97个活动成员与63个保留成员。跟踪191/194调参，其余五段测试。划分是数字顺序加边界缓冲和近似簇隔离，不是用20261022随机划分；20261022是模型训练种子。

## 从原始数据恢复派生输入

以下命令适用于全新克隆并已放入原始数据的目录。已有本地项目包含这些输出时，脚本会拒绝覆盖，应使用新输出版本并同步相应配置；不要删除已完成的结果来强行重跑。

```bash
python -B scripts/verify_raw.py --inventory datasets/raw/inventory.json --ignore-mtime
python -B scripts/audit_data.py --output experiments/preparation/audit_20261022_03 --full-video-decode
python -B scripts/restore_frozen_audit.py
python -B scripts/repair_mot_gt.py --output datasets/annotations/mot_clean_v1
python -B scripts/restore_frozen_audit.py --tracking-gt
python -B scripts/review_detection_coordinates.py --records .cache/frozen_detection_records.json --output experiments/preparation/coordinate_review_v1
python -B scripts/convert_detection.py --records .cache/frozen_detection_records.json --splits datasets/splits/protocol_v2/detection/splits.json --coordinate-convention voc_1based_inclusive --difficult-policy include --coordinate-review experiments/preparation/coordinate_review_v1/review.json --output datasets/annotations/detection_v2_voc
python -B scripts/prepare_mot.py --gt-manifest datasets/annotations/mot_clean_v1/manifest.json --output datasets/processed/mot/clean_v1
```

`restore_frozen_audit.py`在校验原始图像/XML哈希后，仅尝试LF/CRLF两种换行序列，将匹配冻结哈希的审计记录写入忽略的`.cache/`目录。它不改变标注值，也不修改原审计；若还有其他差异则停止，不能绕过哈希检查。

193的ID6共1904条恒定零宽占位记录被排除，197有20条轻微越界框被裁剪；其余身份编号保留。不允许凭空补框或修改源GT。`--tracking-gt`同样仅允许换行差异：先验证全部序列能匹配冻结内容哈希，再调整派生GT的LF/CRLF字节布局，记录manifest备份与变化；不会写入原始GT。重新生成的GT manifest可能因审计时间与环境记录而具有不同哈希，最终评价仍须核验每段视频与派生GT的内容哈希；冻结跟踪划分保存了这些内容哈希。

本流程使用已发布的固定划分，不重新选择测试成员。若要研究新划分，可另行运行`check_content_overlap.py`和`resolve_data_protocol.py`，生成新协议并与本轮结果分开归档。

## 执行新实验

检查数据、环境和权重后，先查看计划：

```bash
.venvs/detection/bin/python -B scripts/run_experiments.py --output experiments/formal/new_run --plan-only
```

确认使用新批次名后，执行：

```bash
mkdir -p experiments/logs
nohup .venvs/detection/bin/python -B -u scripts/run_experiments.py --output experiments/formal/new_run > experiments/logs/new_run.log 2>&1 &
```

控制器分别调用两个环境：六检测器各100轮，每轮验证；按验证AP50:95选择权重和共享检测器；缓存跟踪输入；每跟踪方法3组调参并冻结；最终测试与汇总。测速阶段等待GPU独占，训练可在显存满足条件时启动。异常会停止队列并保存状态，`--resume`需要经过核验的交接记录，不能凭PID猜测恢复。

训练、评价、缓存、跟踪、选参的单独入口分别为`train_detection.py`、`evaluate_detection.py`、`cache_detections.py`、`run_tracking.py`、`tune_tracking.py`和`evaluate_tracking.py`，参数用`--help`查看。

## 读取已有结果

精选成绩和最终参数位于`experiments/results/run_20261022_01/`。它们是已完成实验的快照，不是可写运行目录；完整检查点、检测缓存、逐帧轨迹和日志仍在实验服务器。仓库中出现检查点相对路径或哈希，不表示该权重文件已随Git下载。

比较口径包括固定单种子、同场景留出、统一派生GT、具体跟踪变体、冻结的行人ReID迁移以及排除解码/I/O的速度。不能将小幅差异表述为统计显著，也不能用测试表现再次选择参数后覆盖本轮结果。
