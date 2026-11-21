# D 场景的无人机航测与地面规划

双击 `start_uav_mapping.cmd`。无人机先沿九条有重叠的测线扫描 84 × 60 米的 D 场景，在每个采样点停稳后采集 RGB-D 和同步位姿。再根据初始图，在任务方向附近最多选三处有地面边缘的未知区进行斜视补拍。小车从 GPS 位置读取两版图并规划东侧接近点和北侧观察点。林下测试点用于验证观测不足时的反馈。仿真结束后会打开四联预览图。可用 `--nadir-only` 仅运行初始俯拍，或用 `--profile local` 运行 20 × 20 米的短场景。

```powershell
.venv\Scripts\python.exe tools/run_uav_mapping.py --profile full
.venv\Scripts\python.exe tools/run_uav_mapping.py --profile full --nadir-only
.venv\Scripts\python.exe tools/run_uav_mapping.py --profile local
```

依赖见 `requirements-search.txt`，建议使用 Webots R2025a。可用 `--visible` 显示世界。每次运行的文件都保存在 `runs/uav_mapping_D_.../`。

- `preview.png` 显示航拍合成与实测航迹、地面高程、风险类型和全局路线。绿色是候选通行，暖色是地形危险，灰色是未知。
- `refinement.png` 放大显示同一局部区域的俯拍和补拍结果。青绿色是补拍新增的候选通行格，蓝灰色是有部分地面证据、仍待核实的格子。
- `uav_map_nadir.npz` 是补拍前的基线，`uav_map.npz` 是发布给车端的最终图。包含 ENU 原点、分辨率、时间、状态、高程、坡度、风险类别、观测帧数和规划代价。`states` 的 -1、0、1 分别表示未知、候选通行和障碍。数组的行号由南向北增长，列号由西向东增长。缺少地面证据的高程用 `NaN` 表示。
- `oblique_survey.json` 保存从初始地图选出的补拍区、视点和采样时间。`refined_ground` 标记新增候选通行格，`oblique_ground_quarters` 记录每格中有重复地面证据的四分格数。
- `observations.npz` 保留同步的深度采样、GPS、姿态、RGB 采样和投影参数。`frames/` 保留实际 RGB 图像。投影对齐检查在相反航向的重叠图像中匹配特征，再比较独立投影所得的 ENU 位置。
- `route.json` 记录小车测得的起点、配置的目标、路线以及无路原因。在同一张地图上对照仅考虑距离的 A*，记录两条路径的长度和路点平均坡度。
- `summary.json` 分别汇总有深度回波和确认了地面的格数，并以独立仿真地形为基准检查已估计的地面高程。`map_config.json` 保留当次使用的配置。

### ETH aerial_mapper 正射与高程产品

Webots 采集与车端规划结束后，会自动从本次机载观测导出 `aerial_products/`。已有观测可用 `python tools/aerial_products.py <runs 下的运行目录>` 重新导出。

地面网格向照片反向投影的做法参考 Hinzmann 等人的 *Mapping on the Fly* 和 ETH `aerial_mapper` 的 `ortho-backward-grid`。这里用 Webots RGB-D 深度和姿态取得实测彩色点云。将同格深度点的高位分位数用于可见表面高度，再把每个有效 DSM 格中心投回多帧图像。从深度一致的候选中选择最接近俯视的视点取色，缺少可信观测处保留透明。[ETH 正射方法与源码](https://github.com/ethz-asl/aerial_mapper/tree/master/aerial_mapper_ortho) · [论文](https://demuc.de/papers/hinzmann2017mapping.pdf)

- `orthomosaic.tif` 和 `surface_dsm.tif` 使用 0.125 米栅格，由同一可见表面配准。DSM 包含树冠和屋顶。
- `ground_dem.tif` 和 `traversability.tif` 由车端使用的 0.50 米地图直接导出。地面缺测写为 `-9999`。通行代码 0、1、255 分别表示候选通行、障碍和未知。
- `measured_cloud.ply` 保留体素抽样后的实测 RGB-D 点与颜色，体素边长 0.15 米。`orthomosaic_sources.npz` 记录每个正射格的来源帧和深度差。
- `terrain_layers.png` 把正射图、表面高程与地面高程放在同一视野中。`products.json` 记录格数、点数和选色时的深度一致性。

四层 TIFF 都以北为上，采用同一范围与本地 ENU 米制原点。TIFF 内含像素尺度和控制点，每层附有用于读取位置的 `.tfw`。这些坐标对应 Webots D 场景。车端规划使用证据充分的地面层和通行层，树冠遮挡下的地面继续保持未知。

地图从无人机传感器生成。先在可见裸地选一个参考格，再向相邻且有重复深度支持的连续地面扩展。在全场模式中，估计坡度超过暂定上限 18° 的格子标记为陡坡。近地台阶、落差和粗糙地表分开记录。树冠和屋顶遮挡的低置信区域保留为未知。A* 的通行代价随坡度、近地高差和观测不确定度增加。参考格、部署点和目标在 `config/scenes/uav_mapping_full_D.json` 中定义，阈值需要用车辆通过性实验标定。

补拍阶段用相邻候选通行格的高程拟合局部地面。一个 0.5 米未知格划分成四个 0.25 米子格，每个子格都要从相隔至少 2 米的视点重复看到符合该地面高度的回波。格子还需至少得到三帧支持、通过坡度检查，并排除反复出现的车身高度附近障碍回波和可能的落差。已有通行和障碍判定保留在地图中。检测评估时才读取仿真地形参考。

本演示在生成的 D 派生世界里进行，倒木和碎片固定在初始位置。小车在测区外等待采图，地图发布后在演示起点部署，并停在这个位置做规划。A* 避开未知和障碍并按 0.40 米半径留出车体余量。上述入口保持为航测与全局路线基线；衔接车端闭环行驶、车载 LiDAR/RGB-D 避障和独立真值评测的入口见 [UAV_NAVIGATION.md](UAV_NAVIGATION.md)。
