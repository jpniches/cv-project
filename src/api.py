from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.video_processor import VIDEO_EXTENSIONS, process_video

from pathlib import Path
from uuid import uuid4

from fastapi.concurrency import run_in_threadpool

from src.image_processor import process_image

app = FastAPI(title="Thermal Box Detector")

ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT_DIR / "outputs" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Recebe o vídeo e executa o processamento utilizando YOLO.
@app.post("/videos")
async def upload_video(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    # Valida o formato do vídeo enviado.
    if extension not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Formato de vídeo não suportado.",
        )

    # Salva o vídeo temporariamente para processamento.
    input_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"

    try:
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                destination.write(chunk)

    finally:
        await file.close()

    try:
        # Processa o vídeo e gera o arquivo com as detecções.
        output_path = process_video(input_path)

        # Retorna o vídeo processado para download.
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=f"{Path(filename).stem}_detectado.mp4",
        )

    finally:
        # Remove o arquivo temporário após o processamento.
        input_path.unlink(missing_ok=True)


# Recebe uma imagem e executa a detecção de objetos utilizando YOLO.
@app.post("/images")
async def detect_image(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    # Valida o formato da imagem enviada.
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Formato de imagem não suportado.",
        )

    # Define um caminho temporário para armazenar a imagem.
    input_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"

    try:
        # Salva a imagem enviada para processamento.
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                destination.write(chunk)

        # Impede o processamento de arquivos vazios.
        if input_path.stat().st_size == 0:
            raise HTTPException(
                status_code=400,
                detail="O arquivo está vazio.",
            )

        # Executa a detecção e gera a imagem com os objetos identificados.
        output_path = await run_in_threadpool(
            process_image,
            input_path,
        )

        # Retorna a imagem processada.
        return FileResponse(
            path=output_path,
            media_type="image/jpeg",
            filename=f"{Path(filename).stem}_detectado.jpg",
            content_disposition_type="inline",
        )

    finally:
        # Remove o arquivo temporário após o processamento.
        input_path.unlink(missing_ok=True)
        await file.close()