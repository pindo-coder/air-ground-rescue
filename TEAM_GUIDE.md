# 队友复现与 Git 协作

## 仓库根目录

以本文件所在的 `rescue_vision_lab_flood` 为仓库根目录，其中直接包含 `worlds/`、`protos/`、`controllers/` 和启动脚本。不要把外层日期文件夹或工作区中的其他工程一起上传。

必需文件已经随项目提供：三套 `.wbt` 世界、`.proto`、全部本地纹理、控制器、公共任务配置和评测元数据。`evaluation/` 是运行评测器必需的数据；它与策略输入的分离是接口约定，不应从仓库删除。`preview/` 是场景预览，可直接查看。

`.gitignore` 排除了运行输出 `runs/`、Python 缓存、临时验证世界、Webots 自动缩略图、个人 Python 配置和本机 Webots 路径。现有场景的 `.wbproj` 界面布局保留在仓库中。被忽略的文件仍留在各自电脑上。

本次采用普通 Git，未配置 Git LFS；当前没有超过 50 MiB 的单文件。后续若引入更大的模型或经常更新大型二进制素材，先商量存储方式，再提交。不要把素材压缩包和原始训练数据复制进仓库。

## 首次获取和运行

安装 Git、Webots R2025a 和 Python 3。在 PowerShell 中执行：

```powershell
git clone https://github.com/pindo-coder/webots-vision-rescue.git rescue_vision_lab_flood
cd rescue_vision_lab_flood
```

在 Webots 的 Preferences 中，把 Python command 设置为自己电脑的 Python 解释器。这里使用 Webots 自带的 `controller` 模块，无需通过 pip 安装同名包。静止预览不需要安装第三方 Python 包。

双击 `launch_B.cmd` 打开主场景；`launch_A.cmd`、`launch_C.cmd` 打开对照场景。也可以在 Webots 中选择 File → Open World，打开 `worlds/` 下的相应 `.wbt`，再点击运行。

启动脚本依次检查本地配置、已有 `WEBOTS_EXE`、`WEBOTS_HOME`、默认安装目录和 PATH。若找不到 Webots，复制配置示例后编辑其中的安装目录：

```powershell
Copy-Item webots.local.cmd.example webots.local.cmd
```

`WEBOTS_HOME` 指 Webots 安装根目录；`WEBOTS_EXE` 可直接指向完整可执行文件。`webots.local.cmd` 与 `controllers/` 下的个人 `runtime.ini` 都不会提交。

普通启动默认静止预览。`start_rover_B.cmd` 是包内已有的可选运动演示，需要 `requirements-drive.txt`；它不是你们要训练的机器学习救援算法。不要全局设置 `VISION_DRIVE=1`。

## 生成和验证

运行现有世界无需重新生成。需要修改生成器或跑离线几何检查时，在选定的 Python 环境中执行：

```powershell
python -m pip install -r requirements-build.txt
python -m unittest discover -s tests -v
```

构建依赖固定为本次检查使用的 NumPy 2.3.5、Pillow 13.0.0，本次离线检查使用 Python 3.12.14。建议用独立 Python 环境安装构建依赖；Webots 的控制器解释器可单独配置。

完整重新生成会覆盖世界文件、纹理及评测元数据，应与自己的修改一起提交。配置默认输出 2048 像素，当前成品使用 4096 像素；需要保持同等分辨率时运行：

```powershell
python tools/build_scenarios.py --texture-size 4096
```

可选的 Webots 冒烟检查会打开模拟器，按 A/B/C 各运行 12 个仿真秒，并把本次结果写入 `runs/`。它不测试自主导航成功率：

```powershell
python tools/validate_in_webots.py --webots "<Webots安装目录>/msys64/mingw64/bin/webots.exe"
```

`preview/validation.json` 是随包携带的旧预览记录，不能替代本机重新验证的结果。

## 日常协作

更新之前先保存并关闭 Webots 中的世界，再检查本地改动；否则旧窗口随后保存可能覆盖刚拉取的新文件。工作区干净时更新：

```powershell
git status
git pull --ff-only
```

修改前先与队友约定谁负责哪套世界；生成器与其输出应由同一人一起更新。不要同时手改同一个 `.wbt` 或生成同一组贴图。可为每次修改建立分支：

```powershell
git switch -c scene/your-change
# 修改后检查文件差异，运行相关检查
python tools/restore_public_tasks.py
git diff --check
git diff --stat
python -m unittest discover -s tests -v
git add .
git diff --cached --check
git diff --cached --stat
git commit -m "Describe the scene change"
git push -u origin HEAD
```

新增 Python 依赖时同时更新对应的依赖文件。运行输出不进仓库；需要给队友展示某次结果时，挑选少量图片放入 `preview/`，并写明来源。

Webots 在仿真运行后保存世界，可能把机器人 `customData` 写成实时遥测，覆盖初始相机和任务参数。提交前关闭该世界，再运行 `python tools/restore_public_tasks.py`；此工具只恢复公共任务字段，不改地形、模型或视角。只读检查使用 `python tools/restore_public_tasks.py --check`，离线测试也会检查这一项。恢复后再重新打开世界，避免旧窗口重新覆盖它。

## 仓库与素材许可

共享仓库为 [pindo-coder/webots-vision-rescue](https://github.com/pindo-coder/webots-vision-rescue)，主分支为 `main`。队友从同一个仓库获取更新，无需反复传递整个模型压缩包。

素材出处和各自许可保留在 `MATERIAL_SOURCES.md` 及素材源文件中。本次没有替整个工程添加开源许可证；若计划公开仓库，请先确定队伍代码的授权方式，并核对所用素材的许可条件。
