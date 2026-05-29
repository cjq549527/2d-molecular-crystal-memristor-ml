# 数据来源与字段说明

## 1. 数据集定位

本程序包含两个数据表：

- `data/raw/source_2d_memristor_materials.csv`：源域二维/超薄忆阻器材料数据，用于训练和比较机器学习模型。
- `data/candidate_materials.csv`：目标域二维无机分子晶体候选材料，用于输出实验优先级排序。

由于二维无机分子晶体忆阻器的直接器件数据仍然有限，本程序采用“小样本迁移筛选”的课程报告设定：源域表提供二维材料忆阻器的一般规律，目标域表提供分子晶体的材料描述符、MBE 可行性和 STM/STS 先验。程序输出为实验优先级，不是器件性能定论。

## 2. 输入样例渠道

运行 `python run_pipeline.py` 后，程序会自动生成：

- `data/samples/sample_source_2d_memristor_materials.csv`
- `data/samples/sample_candidate_materials.csv`
- `data/samples/full_candidate_materials.csv`

其中源域样例来自公开二维材料忆阻器文献和二维材料数据库整理；候选材料样例来自公开二维无机分子晶体文献、用户提供的课程材料表和实验先验。

## 3. 关键字段

| 字段 | 含义 | 取值方式 |
| --- | --- | --- |
| `bandgap_eV` | 带隙，单位 eV | 文献值、数据库值或缺失值中位数填补 |
| `defect_trap_score` | 缺陷/陷阱态相关评分 | 根据忆阻器机制文献和材料缺陷敏感性整理 |
| `ion_migration_score` | 离子迁移或空位迁移潜力 | 根据氧化物、硫族化物和卤化物机制先验整理 |
| `air_stability_score` | 空气稳定性 | 文献描述和实验先验等级编码 |
| `thermal_stability_score` | 热稳定性 | 热分析、升华/分解温度或实验描述编码 |
| `fabrication_score` | 常规制备成熟度 | CVD、MBE、机械剥离、热蒸镀等工艺成熟度 |
| `stm_evidence_score` | STM/STS 证据强度 | 是否有原子级形貌、局域态或表面有序证据 |
| `surface_order_score` | 表面有序度 | STM/AFM/TEM 等结构证据编码 |
| `encapsulation_need_score` | 封装需求 | 空气敏感、吸潮或分解风险编码 |
| `sublimation_temp_c` | 升华温度 | 用户提供表格或公开文献 |
| `decomposition_temp_c` | 分解温度 | 用户提供表格或公开文献 |

## 4. 缺失值原则

对公开文献暂未检索到的材料参数，不在程序中伪造精确数值。训练阶段的数值缺失用源域中位数填补，并在报告中说明“不确定性来自材料数据不完整”。目标材料排序中，缺失升华/分解窗口会被写入制备风险项，而不是简单按高热稳定性加分。

## 5. 文献索引

`references.csv` 给出本程序和报告使用的主要文献，包括二维无机分子晶体、二维材料忆阻器、机器学习材料发现、可复现机器学习和代表性机器学习算法。
