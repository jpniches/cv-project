from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.video_processor import VIDEO_EXTENSIONS, process_video


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