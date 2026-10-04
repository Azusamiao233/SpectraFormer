"""用于时间序列缺失值插补的 Transformer 模型。

Lazy exports keep optional plotting dependencies out of headless training runs.
"""

__all__ = ["TransformerImputer", "MSSATransformerImputer"]


def __getattr__(name):
    if name in __all__:
        from .transformer_imputation import MSSATransformerImputer, TransformerImputer

        return {
            "TransformerImputer": TransformerImputer,
            "MSSATransformerImputer": MSSATransformerImputer,
        }[name]
    raise AttributeError(name)
