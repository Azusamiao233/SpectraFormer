"""项目统一入口。

推荐使用 ``python main.py --list-models`` 查看可运行模型，或直接执行
``python main.py`` 打开交互式选择菜单。
"""

from spectraformer.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
