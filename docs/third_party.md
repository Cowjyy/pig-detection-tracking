# 第三方来源与方法变体

上游提交固定在`configs/upstream_sources.json`。`python scripts/bootstrap_sources.py`按该文件获取源码，不提交整个`tools/vendor/`目录。

| 项目 | 用途 | 固定版本与许可 |
|---|---|---|
| MMDetection | 六种检测器训练和评价 | 3.3.0，固定提交；Apache-2.0 |
| BoxMOT | 六种跟踪方法的适配实现 | 12.0.4，固定提交；AGPL-3.0 |
| TrackEval | MOT指标评价 | 固定提交；以其上游LICENSE为准 |
| OSNet | 三种ReID跟踪器的冻结外观模型 | 作者MSMT17 combineall权重，来源及SHA-256见跟踪配置 |

项目链接：

- https://github.com/open-mmlab/mmdetection
- https://github.com/mikel-brostrom/boxmot
- https://github.com/JonathonLuiten/TrackEval
- https://github.com/KaiyangZhou/deep-person-reid

StrongSORT仅使用在线部分，不包含AFLink与GSI。BoostTrack采用无ReID基础配置，不标为BoostTrack++。BoT-SORT、StrongSORT和Deep OC-SORT共用冻结的行人OSNet权重，未在猪只身份数据上训练。对外使用时应保留具体变体和来源说明。
