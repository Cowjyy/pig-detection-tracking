# 检测标注坐标协议

2026-10-22已将本项目实验口径固定为`voc_1based_inclusive`。这是根据现有文件证据采用的标准VOC解释；没有把未知的原始标注工具或历史版本写成已核实事实。

全部1015张图、9735个框都满足1-based坐标范围。xmin=1共8个框，ymin=1共210个框，ymax等于图高共939个框，没有零起点。更强的证据是：所有框的`truncated`值均与LabelImg公开写入器的边界规则完全一致，差异数为0。XML结构、字段顺序及边界行为都与这种常见VOC导出一致。参照[LabelImg VOC写入器](https://github.com/HumanSignal/labelImg/blob/master/libs/pascal_voc_io.py)及[坐标导出实现](https://github.com/HumanSignal/labelImg/blob/master/libs/labelFile.py)。仅凭这些证据不能证明工具历史，因此配置明确保留`original_annotation_tool_verified=false`。

转换规则：COCO的x=xmin−1、y=ymin−1、width=xmax−xmin+1、height=ymax−ymin+1；YOLO标签从这一统一COCO几何计算。所有检测器使用同一份转换。difficult统一include，原始数据中没有difficult对象。

当前转换位于`datasets/annotations/detection_v2_voc/`，训练660图/6287框、验证195图/1897框、测试97图/944框。63张reserve图不进入转换后的活动集合，原始文件完整保留。

证据和复核入口：

- `experiments/preparation/coordinate_review_v1/review.json`：全量边界、truncated规则与假设声明。
- `experiments/preparation/conversion_verification_v1/verification.json`：独立从XML反算COCO坐标并检查YOLO数值。
- 同目录`conversion_preview.jpg`：训练/验证/测试各两张框叠加图。
- 每次转换的`conversion.json`：原始审核记录、划分与坐标证据SHA-256。

这项坐标口径已经固定，允许在明确披露该协议后开始实验。若未来获得标注工具的相反证据，应新增转换版本并重跑受影响实验，不能静默替换现有版本。
