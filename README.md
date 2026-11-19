# Rooster

本仓库包含班表与睡眠事件的本地持久化时间线、班表解析、睡眠生理模型及反应式排程 HTTP 接口。前端位于独立仓库 `D:\Coding\Projects\Coflight-rooster-integration`。

使用 Conda 环境 `205agent`。安装运行依赖后启动：

```powershell
conda activate 205agent
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

班表预览需要配置 `MINIMAX_API_KEY`，用户确认后才会保存至时间线。

## 本地数据库

- 实际使用的数据库：`D:\Coding\Projects\Rooster-main\.timeline_kens\kernel.db`。未设置环境变量时，默认路径为**启动后端时的当前目录**下的 `.timeline_kens/kernel.db`。
- 测评沙盒初始数据库：`D:\Coding\Projects\Rooster-main\.timeline_kens\evaluation-initial-2026-11.db`，包含 Eva 的测评初始日程。模拟实际睡眠不代表真实记录。

`ROOSTER_TIMELINE_DB` 可指定后端使用的数据库。为保留可重复使用的沙盒初始状态，先复制一份，再启动后端：

```powershell
conda activate 205agent
Copy-Item 'D:\Coding\Projects\Rooster-main\.timeline_kens\evaluation-initial-2026-11.db' 'D:\Coding\Projects\Rooster-main\.timeline_kens\evaluation-run.db'
$env:ROOSTER_TIMELINE_DB = 'D:\Coding\Projects\Rooster-main\.timeline_kens\evaluation-run.db'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

同一个 PowerShell 会话中切回实际数据库时，先停止后端，再执行 `Remove-Item Env:ROOSTER_TIMELINE_DB`，并从 `D:\Coding\Projects\Rooster-main` 启动后端。数据库文件为本地数据，不随 Git 提交。

运行测试：

```powershell
conda activate 205agent
pip install -r requirements-dev.txt
$env:ROOSTER_IMPORT_PATH = 'app.invocation'
$env:ROOSTER_RESET_FN = 'app.invocation.reset_isolated_db'
python -m pytest tests -q
```

睡眠模型输出是模型计算结果，不是临床疲劳或适任判断。
