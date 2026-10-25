# 标注处理与内容隔离结果：protocol_v2 / mot_clean_v1

本文保留数据准备阶段的处理记录。正式六检测与六跟踪实验已于2026年10月22日完成；当前执行入口见`docs/reproduction.md`，结果见`docs/results.md`。下文“未启动”等表述指当时的数据准备阶段。

本次依据用户“直接解决”的指示处理现有文件，未启动检测/跟踪实验。所有清洗由Python执行，原始图片、XML、视频和GT完整保留；旧版本清单也保留。

## 已处理的跟踪GT

| 序列 | 原始问题 | 派生处理 | 输出情况 |
|---|---|---|---|
| 193 | ID 6的1904条记录在同一坐标，宽始终为0 | 排除这条恒定零面积占位记录，不填造框、不插值、不重排其余ID | 7个有效ID，13328行 |
| 197 | ID 7有20条框轻微越出画面 | 与1920×1080画面求交，最多允许5像素越界、至少保留90%原面积 | 7个ID，1811帧全部保留 |
| 其余5段 | 无当前已检出的几何错误 | 保留标注内容 | 未作数值修订 |

193已回看首/中/末帧，零宽记录没有形成可观察的独立目标；保留的7个正常框与抽样画面相符。这是对恒定退化占位条目的排除，不是声称人工重新标完1904帧。197已回看异常首帧和越界结束附近画面，保留画面内的可见部分。

派生7段GT共100103行，已逐行重新验证9列格式、正面积、边界、帧号、有效字段和每帧ID唯一性。身份编号原样保留，193没有ID 6属于正常的非连续ID编号。不能将这些结果描述成对全部身份标注的逐帧人工核验。

入口：`datasets/annotations/mot_clean_v1/manifest.json`。`changes.json`包含全部1924条变更的源行号、帧号、ID、原始行、派生行及原因；manifest包含原始/派生GT和视频SHA-256。

后续所有跟踪方法都使用此版本GT，报告中明确写“剔除零面积占位条目并裁剪轻微越界的派生标注”。不得将分数冒充未经处理的原始数据集基准分数。

## 图片与视频内容检查

已将7段视频全部12749帧纳入检索，对1015张检测图片逐一搜索。先在32×18彩色特征上遍历全部视频帧，为每图提取5个最近候选，再用160×90灰度SSIM、全图MAE及猪只框ROI的MAE复核。判定阈值：SSIM≥0.98、全图MAE≤3、ROI MAE≤4。

- 当前未发现满足这些阈值的图片—视频重复匹配。
- 最相似检索候选的SSIM约0.665，明显低于阈值；这只针对当前对齐/缩放搜索的度量。
- 用每段真实首帧执行“缩放到1280×720、JPEG压缩、再检索”的7个正控制均通过，SSIM约0.9995～0.9997。
- 针对旧划分筛查的近邻及dHash候选，未发现满足上述结构/ROI阈值的跨集合直接重复。

完整证据：`experiments/preparation/content_overlap_v2/{summary,retrieval,control_checks}.json`，可复核每张图片的候选帧号与分值。`thumbnail_cache.npz`保留对应缩略图。

先前的`content_overlap_v1`为未通过自检的尝试，只留有缓存，不得作为结论依据；v2将正控制改成与真实数据一致的“高分辨率JPEG后缩略”流程，完成检索与自检。

**没有直接匹配不等于证明不同来源。**现有图片缺乏采集视频ID、时间戳、动物ID和采集批次，算法不能凭空恢复这些事实；裁剪、几何变换或强光照变化也可能漏检。因此当前协议明确限定为“现有同场景数据的留出评估”，不能宣称跨栏舍、跨动物或跨采集来源的独立泛化。这一事实已写入配置，而非将unknown改成true来解除提示。

## 当前检测划分

| 集合 | 图片/XML对数 | 用途 |
|---|---:|---|
| train | 660 | 检测训练 |
| val | 195 | 模型、阈值选择 |
| test | 97 | 冻结设置后的检测评价 |
| reserve | 63 | 只保留，不参与训练、调参或评分 |

流程：先从数字编号顺序预留两处各20张的边界缓冲，剩余975张按约7:2:1连续分配；再对全部图片计算dHash≤4的连通簇。若一个簇跨集合，则保留优先级test>val>train的成员，其余转入reserve，不把测试成员移入训练。

最终40张为边界缓冲、23张为跨集合近似簇的保守隔离；共63张均在原目录保留，不是删除。dHash会因共同背景误报，隔离策略有意保守，不把这23张称为已证实的重复图片。

当前三个活动集合之间dHash≤4候选对数为0。它只是对已定义启发式的检查结果，不是完整的时间独立或来源独立证明。编号相邻的20张不是已知20秒或20帧，不能在报告中捏造采样间隔。

入口：`datasets/splits/protocol_v2/detection/splits.json`；同目录train/val/test/reserve TXT与CSV均已生成。所有1015对在活动集合与reserve中恰好出现一次。旧711/203/101的initial_v1已由本版本替代。

## 当前跟踪划分

调参191/194，测试192/193/195/196/197，整段视频不跨集合，成员与v1相同。GT使用clean_v1，原来193/197的几何阻塞已解除。入口：`datasets/splits/protocol_v2/tracking/splits.json`，明确区分source_gt与当前gt。

已生成元数据与GT副本：`datasets/processed/mot/clean_v1/`，未抽取全部视频帧。其READY内容为metadata_only，不能误当成含img1目录的图像输入；可用原视频流，或在服务器上再抽帧。

## 可复现命令

```bash
python -B scripts/check_content_overlap.py --output experiments/preparation/new_content_check
python -B scripts/repair_mot_gt.py --output datasets/annotations/mot_clean_v2
# 以下参数应指向本次真实生成的报告/GT，不要混用旧版本：
python -B scripts/resolve_data_protocol.py --overlap experiments/preparation/new_content_check/summary.json --gt-manifest datasets/annotations/mot_clean_v2/manifest.json --output datasets/splits/new_protocol
python -B scripts/prepare_mot.py --gt-manifest datasets/annotations/mot_clean_v2/manifest.json --output datasets/processed/mot/new_metadata
# 需要完整图片帧时在服务器执行，输出用另一个新目录：
python -B scripts/prepare_mot.py --gt-manifest datasets/annotations/mot_clean_v2/manifest.json --output datasets/processed/mot/new_frames --extract-frames
python -B scripts/verify_raw.py --inventory experiments/preparation/audit_20261022_03/raw_inventory.json
```

准备工具只解决上述数据协议和几何问题。检测VOC坐标约定、服务器依赖及模型适配仍按原计划处理，不在此次清洗中猜测或启动训练。

## 最终核验

`scripts/verify_resolved_data.py`独立核对了所有GT变更、导出副本、活动集合与保留集的完整覆盖，并重新计算跨集合dHash和原始文件清单。结果保存在`experiments/preparation/resolution_verification_v1/results.json`，20项单元检查全部通过，2051个原始文件的路径、大小、mtime和SHA-256均与基线一致。193/197的清洗后首帧叠框图同目录保存并已查看。

```bash
python -B scripts/verify_resolved_data.py --output experiments/preparation/new_resolution_verification
```
