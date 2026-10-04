"""奇异谱分析（SSA/MSSA）实现。

Public objects are loaded lazily so importing a non-plotting MSSA implementation
does not require Matplotlib merely because this package was initialized.
"""

__all__ = ["MSSA", "setup_chinese_fonts"]


def __getattr__(name):
    if name in __all__:
        from .mssa1 import MSSA, setup_chinese_fonts

        return {"MSSA": MSSA, "setup_chinese_fonts": setup_chinese_fonts}[name]
    raise AttributeError(name)
