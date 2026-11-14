# 搜救模块集成验证（2026-11-12）

基于源提交9a6a1e702597dfb485fb660df64544e2edac87e3，原纯视觉演示与新增LiDAR/RGB-D模块分开运行。

## 实际预检结果

- 37项离线验证全部通过，包含原有18项与新增19项算法/评价契约检查。

- A/B/C原地图的新增控制器、传感器和YOLO接口分别通过3秒启动检查。初始位置目标不可见，均无有效报告；这属于删失的启动检查，不是搜索效果实验。
- frontier、weighted、active_only、schedule_only分别通过独立平地夹具的8秒运行检查，未解释为性能比较。
- full在独立平地夹具45秒预检中完成1个有效报告、1次按期有效投送，0个误报、0次碰撞。首次有效报告10.0秒，有效投送18.1秒，唯一匹配报告定位误差约0.288m。
- 平地预检使用surrogate_rgbd颜色/尺寸候选器；它不是人员识别准确率实验。共有3个人员，总任务成功标志为false。不能将单次预检视作性能提升或把未服务人员删出分母。
- 同一平地夹具的较早开发运行曾出现未完成投送和定位误差偏大；已保留原日志，没有把它们混入同版本对比。修复包括观察位姿到达裕量，以及投送前的近距离相对观测复核。

## 证据位置（本机runs不提交Git）

- full有效投送运行：runs/search_smoke_full_surrogate_rgbd_20261112_233900_29296/
- 原地图启动矩阵：runs/search_matrix_20261112_234212.json
- 基线启动矩阵：runs/search_matrix_20261112_234350.json
- 全部单元验证：runs/search_validation/unit_tests.log
- 原始逐次指标：runs/search_metrics.csv（运行tools/analyze_search_runs.py重新导出）

运行manifest保存代码、世界、素材、模型及依赖哈希。正式比较仍需独立布局、标定的检测器/定位误差模型、统一检测条件和足够仿真时长。具体运行见SEARCH_ALGORITHMS.md，研究假设见SEARCH_RESEARCH_PROTOCOL.md。
