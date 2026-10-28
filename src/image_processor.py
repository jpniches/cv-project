# Serviço responsável pelo processamento de imagens utilizando YOLO.
# Realiza a detecção de objetos e salva a imagem com as identificações.

from pathlib import Path

import cv2
from ultralytics import YOLO

from src.utils.model import get_model

# Diretório destinado ao armazenamento das imagens processadas.
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs" / "predictions"

# Nível mínimo de confiança para considerar uma detecção.
CONFIDENCE = 0.25

# Processa uma imagem utilizando o modelo de detecção YOLO.
def process_image(
    input_path: str | Path,
    model: YOLO | None = None,
) -> Path:
    input_path = Path(input_path).resolve()

    # Valida a existência e o formato da imagem.
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Imagem não encontrada: {input_path}"
        )

    if input_path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }:
        raise ValueError(
            f"Formato de imagem não suportado: {input_path.suffix}"
        )

    # Carrega o modelo e executa a detecção dos objetos.
    model = model or get_model()

    results = model.predict(
        source=str(input_path),
        conf=CONFIDENCE,
        verbose=False,
    )

    # Gera a imagem com as detecções identificadas.
    annotated_image = results[0].plot()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define o caminho para salvar a imagem processada.
    output_path = OUTPUT_DIR / f"{input_path.stem}_detectado.jpg"

    # Salva a imagem e verifica se a operação foi concluída.
    if not cv2.imwrite(str(output_path), annotated_image):
        raise RuntimeError(
            f"Não foi possível salvar a imagem: {output_path}"
        )

    # Retorna o caminho da imagem processada.
    return output_path