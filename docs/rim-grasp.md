# 薄罐沿抓取：已有证据与本项目范围

薄口沿抓取涉及两个连续环节：手指跨在开口内外两侧形成有效接触，以及在适当接触力下抬起并稳定持物。已有真机研究分别考察了夹沿、物体变形与滑移反馈：

| 一手研究 | 真机上具体看到了什么 | 和 SO-101 的关系 |
|---|---|---|
| Herland、Bach、Misimi，[6-DoF Closed-Loop Grasping with Reinforcement Learning](https://doi.org/10.1109/ICRA57147.2024.10610080)，ICRA 2024，[作者全文](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/3180347/Herland_etal_202494881.pdf?sequence=1)，见 *Challenging Objects* | 对高圆柱形物体，作者未观察到横向包住主体的抓法；蛋白摇摇杯的五次成功抓取均发生在薄口沿。汤碗则可从碗沿稳定抓起。 | 展示了由物体几何决定的夹沿策略，并提供真机案例。 |
| Shahidzadeh、Caddeo 等，[FeelAnyForce](https://www.prg.cs.umd.edu/FeelAnyForce)，ICRA 2025，[论文 §IV-C、Fig. 4](https://prg.cs.umd.edu/research/FeelAnyForce_files/FeelAnyForce.pdf) | 用两枚 GelSight Mini 接触塑料杯口，以杯口圆形到椭圆形的变化估计形变，并按两档目标力重复测量。 | 量化接触力与杯口形变之间的关系；本项目使用 RGB 相机，不包含该触觉测量。 |
| [Robot Intelligent Grasp of Unknown Objects Based on Multi-Sensor Information](https://www.mdpi.com/1424-8220/19/7/1595)，2019，Fig. 33–35 | 在三指夹持的塑料杯中逐渐加米，以滑移反馈补力；实验报告了滑移抑制和重载下的杯体变形。 | 展示滑移反馈与结构形变的并存，说明抓取评估需同时记录持物和物体状态。 |

## 本项目的仿真对象

仿真罐体高 90 mm、外径 100 mm、质量 80 g。底面与 32 段、壁厚 2 mm 的碰撞几何围成真实开口，[场景实现](../sim/grasp_sweep.py)据此评估接近、沿口接触、起抬和重抓。刚体模型不包含罐壁压缩、指尖力测量、永久形变或材料相关摩擦；主任务的布面是桌面上的刚性彩色区域。

本项目聚焦于**单卡 π0.5 后训练下的双 RGB 薄罐沿抓取、持罐交接和整底布面放置**。真机评估将记录罐体材质、实测尺寸、滑移、放置稳定性与夹痕，并与仿真指标分开报告。
