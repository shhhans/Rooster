# Rooster

本仓库包含班表与睡眠事件的本地持久化时间线、班表预览解析、睡眠生理模型及规划代码。前端、独立增量规划服务和 Playground 暂不包含在内；规划模块目前以 Python 接口提供，尚未通过本仓库的 HTTP 服务接入完整规划流程。

使用 Conda 环境 `205agent`。安装运行依赖后启动：

```powershell
conda activate 205agent
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

班表预览需要配置 `MINIMAX_API_KEY`，其结果不会自动写入时间线。时间线默认数据库为 `.timeline_kens/kernel.db`，可用 `ROOSTER_TIMELINE_DB` 指定路径。

运行测试：

```powershell
conda activate 205agent
pip install -r requirements-dev.txt
$env:ROOSTER_IMPORT_PATH = 'app.invocation'
$env:ROOSTER_RESET_FN = 'app.invocation.reset_isolated_db'
python -m pytest tests -q
```

睡眠模型输出是模型计算结果，不是临床疲劳或适任判断。
