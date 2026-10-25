# 模型权重

权重保留在本地或实验服务器，不上传Git。

- `pretrained/`：六种检测器的COCO预训练权重。
- `reid/`：冻结的OSNet x0.25 MSMT17 combineall权重。
- 正式训练得到的检查点由各运行目录保存，当前不包含在代码仓库中。

下载命令、来源及固定版本见[复现说明](../docs/reproduction.md)和`configs/`。OSNet预期SHA-256记录在`configs/tracking/experiments.json`，下载后必须校验。
