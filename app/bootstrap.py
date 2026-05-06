"""把本地依赖目录加入 Python 搜索路径。

这个学习项目支持两种依赖安装方式：

1) 常规虚拟环境（.venv）
2) `pip install --target .packages -r requirements.txt`（仓库自包含，不污染全局）

为了同时支持两种方式，这里在导入 `app.*` 时会优先把 `.packages` 加到 `sys.path`。
如果 `.packages` 不存在，逻辑不会影响正常的 venv 解析。
"""

from __future__ import annotations

import sys
from pathlib import Path


def configure_local_packages() -> None:
    """优先使用项目内的 `.packages` 目录。

    这样做有两个好处：
    1. 学习项目自包含，不污染全局环境。
    2. 测试和运行都能拿到同一套依赖版本。
    """

    # app/bootstrap.py -> app/ -> project root
    project_root = Path(__file__).resolve().parents[1]
    vendor_dir = project_root / ".packages"
    vendor_path = str(vendor_dir)

    if vendor_dir.exists() and vendor_path not in sys.path:
        # 插到最前面，确保本地 vendor 依赖优先生效。
        sys.path.insert(0, vendor_path)

