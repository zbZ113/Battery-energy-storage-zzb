"""Safe adapters for externally sourced battery datasets."""

from quanxin_life.data.adapters.matr import MatrCell, load_matr_batch

__all__ = ["MatrCell", "load_matr_batch"]
