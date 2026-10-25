# 正式结果快照 run_20261022_01

六种检测器各训练100轮，固定种子20261022。六种跟踪器使用验证集选择的YOLOX-S第5轮权重，在191/194上各调参3次，冻结后评价5段测试视频。批次于2026年10月22日完成83/83个阶段。

本目录是正式运行输出的精选副本，不包含原始数据、逐帧轨迹、日志、模型权重或Word。

| 文件 | 含义 |
|---|---|
| detection_comparison.csv | 六检测器的最终测试结果 |
| detection_per_seed.csv | 固定单种子检测成绩 |
| tracking_comparison.csv | 五段视频合并指标与计算速度 |
| tracking_per_sequence.csv | 全部30行逐视频成绩 |
| results.json | 原始完整精度的汇总和选模记录 |
| detector_selection.csv | 验证选模、最佳轮次及权重相对路径 |
| tracking_tuning.csv | 全部18组调参成绩和胜出标记 |
| tracking_decomposition.csv | 从TrackEval COMBINED的AUC字段得到的DetA、AssA与Frag |
| frozen_parameters/ | 六跟踪器最终冻结参数、输入哈希及选择依据 |
| environment/ | 实际环境快照，editable路径改为项目相对路径 |
| figures/ | 检测与跟踪的精度速度散点图 |
| export_manifest.json | 来源文件和导出文件的SHA-256 |

参数文件引用的完整正式运行目录和检查点仍保留在实验服务器，本仓库不含这些大文件。按原有路径说明或获取对应文件后使用；不能将路径存在于JSON视为权重已经上传。

总体跟踪指标来自TrackEval合并计算，不对逐视频百分比取简单平均。合成FPS排除视频解码和文件I/O。仅单种子、同场景留出，不报告显著性或跨来源泛化。
