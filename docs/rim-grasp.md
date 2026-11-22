# 夹一片薄罐壁真的难吗？

**有可靠的实验说明它值得单独研究，但没有理由说“机器人一般抓不起来”。** 要分开看两个问题：手指有没有跨在开口的内、外两侧；跨对之后，能不能在不滑、不伤物的条件下提起来。

| 一手研究 | 真机上具体看到了什么 | 和 SO-101 的关系 |
|---|---|---|
| Herland、Bach、Misimi，[6-DoF Closed-Loop Grasping with Reinforcement Learning](https://doi.org/10.1109/ICRA57147.2024.10610080)，ICRA 2024，[作者全文](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/3180347/Herland_etal_202494881.pdf?sequence=1)，见 *Challenging Objects* | 对高圆柱形物体，作者没观察到横向包住主体的抓法；蛋白摇摇杯成功的五次都夹在薄口沿。汤碗则能稳定地从碗沿抓起。 | 夹沿是有实机先例的抓法。容器几何会改变可用策略；该结果不是 SO-101 的基准。 |
| Shahidzadeh、Caddeo 等，[FeelAnyForce](https://www.prg.cs.umd.edu/FeelAnyForce)，ICRA 2025，[论文 §IV-C、Fig. 4](https://prg.cs.umd.edu/research/FeelAnyForce_files/FeelAnyForce.pdf) | 用两枚 GelSight Mini 接触塑料杯口，把杯口从圆形变为椭圆形的程度作为形变读数；按两档目标力、重复十次测量。 | 即使都能接触到杯沿，实际施力也会改变杯形。该工作的触觉方案不是我们的设备配置。 |
| [Robot Intelligent Grasp of Unknown Objects Based on Multi-Sensor Information](https://www.mdpi.com/1424-8220/19/7/1595)，2019，Fig. 33–35 | 在三指夹住的塑料杯中逐渐加米，用滑移反馈补力；能抑制滑动，但重载下也观察到了杯体变形。 | 防止滑落与避免压变形是两个验收指标。材料和负载变了，不能把别人的指力数值搬过来用。 |

**本项目检验到哪一步？** 仿真对象是 90 mm 高、100 mm 外径、80 g 的刚体。开口是真的：底面和 32 段厚 2 mm 的壁各有碰撞形状，中间不是实心圆柱。[场景实现](../sim/grasp_sweep.py)。这让我们能够比较抓空、碰偏、起抬和重抓。**它不会被夹瘪**；也没有测指尖力、罐壁的永久变形或不同材质的摩擦。布在主任务中也是刚性桌面上的彩色区域，不模拟褶皱。

因此，项目把贡献写成“**在单卡和双 RGB 条件下，分开验证薄罐沿抓取恢复和整底放置**”，不以“首次夹薄壁”或“解决脆弱物体力控”为卖点。真机实验会额外记下罐的材质、实测尺寸、是否滑脱及夹痕；一段成功视频只代表它展示的那一次。
