# As imagens do CD foram tiradas do iPhone, ou seja, ficou com sua extensão padrão, esse arquivo faz a transformção
# de arquivos com extensão .HEIC para .jpg, para que as imagens pudessem ser passadas para o serviço de anotações 
# de imagem

from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

# Registro de suporte para arquivos .HEIC
register_heif_opener()

# Definições de paths para buscar as imagens
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "products"
OUTPUT_DIR = ROOT_DIR / "converted_products"

EXTENSIONS = {".heic", ".heif"}


def main():
    # Cria o diretório de saída caso ele ainda não exista.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Percorre o diretório de entrada e todos os seus subdiretórios,
    # selecionando apenas arquivos com as extensões suportadas.
    files = [
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]

    converted = 0
    errors = 0


    for file in files:
        relative_path = file.relative_to(INPUT_DIR)
        output_path = (OUTPUT_DIR / relative_path).with_suffix(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Evita converter novamente arquivos já existentes em uma possível interrupção do script
        if output_path.exists():
            print(f"Arquivo já existente {relative_path}")
            continue

            
        try:
            # Abre a imagem original e converte seus pixels para RGB,
            # garantindo compatibilidade com a gravação em JPEG.
            with Image.open(file) as image:
                image = image.convert("RGB")
                image.save(output_path, "JPEG", quality=95)

            converted += 1
            print(f"[OK] {relative_path}")


        except Exception as error:
            errors += 1
            print(f"Erro ao converter arquivo: {file.name}: {type(error).__name__}: {error}")

    print("\n=== Resumo ===")
    print(f"Encontradas: {len(files)}")
    print(f"Convertidas: {converted}")
    print(f"Erros: {errors}")
    print(f"Destino: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()