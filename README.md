# 猪只目标检测与多目标跟踪对比实验

比较六种目标检测器及六种在线多目标跟踪方法。正式实验已于2026年10月22日在RTX3090 24GB上完成，固定单种子20261022，六检测器各100轮，83个队列阶段全部完成。

本仓库包含实验代码、配置、正式数据划分和精选结果。原始数据、派生标签、模型权重、缓存、完整日志与Word汇报文档保留在本地或实验服务器，不随Git上传。

## 阅读入口

- [复现说明与执行命令](docs/reproduction.md)
- [检测和跟踪结果分析](docs/results.md)
- [正式结果表及冻结参数](experiments/results/run_20261022_01/README.md)
- [正式数据划分](datasets/splits/README.md)
- [标注修订和内容隔离说明](docs/data_resolution.md)
- [VOC坐标约定](docs/coordinate_protocol.md)
- [第三方来源和方法变体](docs/third_party.md)
- [仓库上传内容范围](docs/repository_contents.md)

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── configs/              数据与实验配置、上游提交、环境约束和快照
├── datasets/
│   ├── raw/              来源说明及原始文件哈希清单，不含数据本体
│   ├── processed/        派生数据说明，实际文件被忽略
│   ├── annotations/      派生标注说明，实际标签被忽略
│   └── splits/           protocol_v2正式JSON、CSV、TXT清单
├── models/               检测与跟踪适配接口
├── losses/               自定义损失预留目录
├── tools/                路径保护、数据读取、哈希与运行工具
├── scripts/              数据准备、训练、测试、跟踪、调参与汇总入口
├── experiments/
│   └── results/          精选正式结果、参数与图表
├── checkpoints/          权重获取说明，实际权重被忽略
├── docs/                 复现、数据协议、结果分析与来源说明
└── tests/                坐标、几何、划分和实验计划检查
```

原始数据取得授权后，仍放在项目根目录的`pig-objection detection/`与`PIG_MOT_TEST_DATA/`。不需要复制到`datasets/raw/`；代码使用项目相对路径。所有数据处理通过Python生成派生文件，拒绝写入原始数据目录。

## 实验范围

- 检测：Faster R-CNN、RetinaNet、FCOS、YOLOX-S、RTMDet-S、Deformable DETR。
- 跟踪：ByteTrack、BoT-SORT、OC-SORT、StrongSORT在线版、Deep OC-SORT、BoostTrack无ReID基础版。
- 检测训练660图、验证195图、测试97图，另有63图保留。
- 跟踪191/194用于每方法3组参数选择；192/193/195/196/197用于冻结后的最终评价。
- 共享YOLOX-S权重按验证集AP50:95选择，不以测试集选择检测器或回调跟踪参数。
- 测试中YOLOX-S的AP50:95为75.0%；BoT-SORT的HOTA为77.88%、IDF1为98.06%。完整比较见结果页。

## 环境与检查

数据准备依赖见`requirements.txt`；它不代替训练环境。正式检测环境为Python 3.10、PyTorch 2.1.0+cu121、MMCV 2.1.0、MMDetection 3.3.0；跟踪环境为Python 3.10、PyTorch 2.2.2+cu121和BoxMOT 12.0.4。精确依赖快照与约束在`configs/environments/`。

```bash
python -B -m unittest discover -s tests -v
python -B scripts/verify_repository_release.py
```

完整实验控制器面向Linux和NVIDIA CUDA服务器，使用项目内两个独立venv；Windows可进行不依赖GPU的数据与代码检查。详见复现说明。

## 结果解释

本轮只运行一个固定种子，不提供跨种子标准差或显著性判断。数据来源独立性未确认，结果属于同场景留出评价。跟踪统一使用修订后的mot_clean_v1 GT；193零宽占位轨迹的排除与197轻微越界框的裁剪已披露。算法均为固定版本适配比较，并非各原论文完整训练配方的复现。
