# Arquivo usado para expor o modelo para o resto dos arquivos da api
from functools import lru_cache
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT_DIR / "models" / "best.pt"


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))
