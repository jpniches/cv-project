# Consolida os exports de anotações do CVAT em um dataset YOLO.
# Valida as imagens e separa as amostras em treino e validação.
# Gera o arquivo data.yaml utilizado no treinamento do detector.

from pathlib import Path
import random
import shutil

# Diretório raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent

# Definições dos caminhos das imagens
IMAGES_DIR = ROOT_DIR / "produtos_convertidos"
PART_1 = ROOT_DIR / "datasets" / "product_images_part_1"
PART_2 = ROOT_DIR / "datasets" / "product_images_part_2"

# Destino do dataset
OUTPUT_DIR = ROOT_DIR / "data" / "detector"

# Mapeamento dos IDs de classe utilizados pelo modelo (Anotações feitas no CVAT)
CLASS_NAMES = {
    0: "caixa-preta",
    1: "caixa-verde",
    2: "caixa-azul",
    3: "caixa-cinza",
    4: "caixa-vermelha",
    5: "caixa-fechada",
}

# IDs permitidos em cada export do CVAT
# Foi separado por parte 1 e parte 2 porque o CVAT não permitia exportar mais que 200 arquivos por vez
SOURCES = [
    (PART_1, {0, 1, 2}),
    (PART_2, {3, 4, 5}),
]

VAL_RATIO = 0.20
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


# Indexa as imagens pelo nome para localizar os arquivos dos rótulos.
def find_images():
    index = {}

    for path in IMAGES_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem.lower(), []).append(path)

    return index


# Valida os rótulos e associa cada anotação à sua imagem.
def collect_samples(image_index):

    # As anotações foram enviadas como coordenadas para cada imagem, pois a 
    # cobertura gratuita da plataforma não permitia exportar as imagens  
    samples = []
    errors = []

    for source_dir, allowed_classes in SOURCES:
        labels_dir = source_dir / "labels" / "train"

        if not labels_dir.is_dir():
            errors.append(f"Pasta de anotações não encontrada: {labels_dir}")
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            matches = image_index.get(label_path.stem.lower(), [])

            if len(matches) == 0:
                errors.append(
                    f"Imagem não encontrada para anotação: {label_path.name}"
                )
                continue

            # Impede associar um rótulo à imagem errada caso haja nomes repetidos. 
            if len(matches) > 1:
                errors.append(
                    f"Nome de imagem duplicado: {label_path.stem} -> "
                    + ", ".join(str(path) for path in matches)
                )
                continue

            image_path = matches[0]
            label_lines = []

            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()

                for line_number, line in enumerate(lines, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    # Cada anotação YOLO contém o ID da classe e quatro coordenadas. 
                    # Como a imagem é um retângulo, cada um é um ponto a ser fechado
                    if len(parts) != 5:
                        raise ValueError(
                            f"Linha {line_number}: esperado ID + 4 coordenadas"
                        )

                    class_id = int(parts[0])

                    if class_id not in allowed_classes:
                        raise ValueError(
                            f"ID {class_id} não permitido neste export; "
                            f"esperado um de {sorted(allowed_classes)}"
                        )

                    if class_id not in CLASS_NAMES:
                        raise ValueError(f"ID de classe desconhecido: {class_id}")

                    coords = [float(value) for value in parts[1:]]

                    # As coordenadas YOLO devem estar normalizadas entre 0 e 1.
                    if not all(0 <= value <= 1 for value in coords):
                        raise ValueError(
                            f"Linha {line_number}: coordenadas fora de [0, 1]"
                        )

                    label_lines.append(
                        f"{class_id} " + " ".join(parts[1:])
                    )

            except (ValueError, OSError) as exc:
                errors.append(f"{label_path}: {exc}")
                continue

            samples.append({
                "image": image_path,
                "name": image_path.name,
                "label_name": image_path.stem + ".txt",
                "label_lines": label_lines,
                "source": source_dir.name,
            })

    # Evita colisões de nomes no dataset final, independentemente da extensão.
    names = [sample["name"].lower() for sample in samples]

    if len(names) != len(set(names)):
        errors.append("Há nomes de imagens repetidos entre os exports.")

    return samples, errors

# Valida, divide e grava o dataset utilizado pelo detector.
def main():

    if not IMAGES_DIR.is_dir():
        raise SystemExit(f"Pasta de imagens não encontrada: {IMAGES_DIR}")

    for source_dir, _ in SOURCES:
        if not source_dir.is_dir():
            raise SystemExit(f"Export não encontrado: {source_dir}")

    image_index = find_images()
    print(f"Imagens JPG encontradas: {sum(map(len, image_index.values()))}")

    samples, errors = collect_samples(image_index)

    # Não cria o dataset enquanto houver problemas nas imagens ou anotações.
    if errors:
        print("\nForam encontrados problemas. Nenhum dataset foi criado:\n")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(
            f"\nCorrija os problemas acima. Total: {len(errors)}."
        )

    if not samples:
        raise SystemExit("Nenhuma amostra válida foi encontrada.")

    # Embaralha as amostras de forma reproduzível e separa treino e validação.
    random.Random(SEED).shuffle(samples)

    val_count = max(1, round(len(samples) * VAL_RATIO))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    # Evita misturar a nova consolidação com arquivos de uma execução anterior.
    if OUTPUT_DIR.exists():
        raise SystemExit(
            f"O destino já existe: {OUTPUT_DIR}\n"
            "Renomeie ou remova o destino antigo manualmente após conferir "
            "se precisa preservar seu conteúdo. Os exports originais "
            "não serão alterados."
        )

    try:
        for split, split_samples in (
            ("train", train_samples),
            ("val", val_samples),
        ):
            images_out = OUTPUT_DIR / "images" / split
            labels_out = OUTPUT_DIR / "labels" / split

            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)

            for sample in split_samples:
                shutil.copy2(
                    sample["image"],
                    images_out / sample["name"],
                )

                content = "\n".join(sample["label_lines"])
                if content:
                    content += "\n"

                (labels_out / sample["label_name"]).write_text(
                    content,
                    encoding="utf-8",
                )

        # Gera a configuração que o Ultralytics utiliza para localizar
        # as imagens de treino, validação e os nomes das classes.
        yaml_content = (
            f"path: {OUTPUT_DIR.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            + "".join(
                f"  {class_id}: {name}\n"
                for class_id, name in CLASS_NAMES.items()
            )
        )

        (OUTPUT_DIR / "data.yaml").write_text(
            yaml_content,
            encoding="utf-8",
        )

    except Exception:
        # Remove a saída parcial para evitar deixar um dataset incompleto.
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        raise

    print("\nDataset criado com sucesso!")
    print(f"Total: {len(samples)} imagens")
    print(f"Treino: {len(train_samples)}")
    print(f"Validação: {len(val_samples)}")
    print(f"Destino: {OUTPUT_DIR}")
    print("\nClasses:")

    for class_id, name in CLASS_NAMES.items():
        print(f"  {class_id}: {name}")


if __name__ == "__main__":
    main()