"""应用包入口。

这个包的一个“学习友好”设计点是：支持把依赖装进仓库内的 `.packages/`。

因此只要有任何 `app.*` 模块被导入，就会先执行：
- `configure_local_packages()`：把 `.packages` 插入到 `sys.path` 前面

这能保证：
- 仓库复制到别的机器也能按同一套依赖运行
- CI/测试环境更容易复现

如果你只使用 `.venv`，且不存在 `.packages`，这段逻辑不会产生副作用。
"""

from .bootstrap import configure_local_packages

configure_local_packages()

