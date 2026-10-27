# Arquivo para treinamento do modelo, resultado final do modelo é uma IA com 
# 90% de precisão, resultado bom para a quantidade de fotos e pensando na variância de 
# luminosidade que temos.

from pathlib import Path
import shutil

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_YAML = ROOT_DIR / "data" / "detector" / "data.yaml"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_OUTPUT = ROOT_DIR / "models" / "best.pt"

# Configurações do treinamento
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 4
PATIENCE = 15


def main():
    # Verifica se o dataset foi preparado antes de iniciar o treinamento.
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {DATASET_YAML}\n"
            "Execute primeiro src/prepare_dataset.py."
        )

    # Carrega o modelo YOLO11n com pesos pré-treinados.
    model = YOLO("yolo11n.pt")

    # Treina o detector com as imagens e configurações definidas.
    model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        project=str(OUTPUT_DIR),
        name="detector-caixas",
        exist_ok=False,
        workers=0,
        seed=42,
        pretrained=True,
        plots=True,
        save=True,
        verbose=True,
    )

    # Localiza o melhor modelo gerado durante o treinamento.
    best_model = Path(model.trainer.best)

    if not best_model.is_file():
        raise FileNotFoundError(
            f"Melhor modelo não encontrado: {best_model}"
        )

    # Copia o melhor modelo para o diretório usado na inferência.
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, MODEL_OUTPUT)

    print("\nTreinamento concluído!")
    print(f"Melhor modelo da execução: {best_model}")
    print(f"Modelo disponível para inferência: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()