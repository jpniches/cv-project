# API responsável pelo processamento de imagens e vídeos utilizando YOLO.
# Gerencia uploads, tarefas, rastreamento e disponibilização dos resultados.

import asyncio
import logging
import uuid

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import cv2

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketDisconnect

from src.utils.model import get_model
from src.image_processor import process_image
from src.video_processor import (
    VIDEO_EXTENSIONS,
    empty_tracking_summary,
    process_video,
)


# Inicializa a aplicação FastAPI.
app = FastAPI(title="Thermal Box Detector")


# Permite a comunicação entre o frontend e a API.
# Apenas para rotas locais 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Diretórios utilizados para armazenar os arquivos enviados e processados.
ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT_DIR / "outputs" / "uploads"
OUTPUT_DIR = ROOT_DIR / "outputs" / "predictions"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Limita o processamento a um vídeo por vez.
executor = ThreadPoolExecutor(max_workers=1)

# Armazena as tarefas e controla o acesso aos recursos compartilhados.
jobs: dict[str, dict] = {}
jobs_lock = Lock()
model_lock = Lock()

logger = logging.getLogger(__name__)


# Atualiza a etapa atual do processamento.
def update_stage(job_id: str, stage: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id)

        if job and job["status"] == "processing":
            job["stage"] = stage


# Atualiza o progresso com base na quantidade de frames processados.
def update_progress(job_id: str, current: int, total: int) -> None:
    percent = min(99, int(current / total * 100)) if total > 0 else 0

    with jobs_lock:
        job = jobs.get(job_id)

        if job and job["status"] == "processing":
            job["frames_processed"] = current
            job["total_frames"] = total
            job["progress"] = percent


# Atualiza os objetos rastreados e o resumo de contagem por classe.
def update_tracking(
    job_id: str,
    boxes: list[dict],
    summary: dict,
) -> None:
    with jobs_lock:
        job = jobs.get(job_id)

        if job and job["status"] == "processing":
            job["tracked_boxes"] = boxes
            job["tracking_summary"] = summary


# Prepara e armazena o frame mais recente para transmissão via WebSocket.
def publish_frame(job_id: str, frame) -> None:
    height, width = frame.shape[:2]
    max_width = 720

    # Reduz a resolução para otimizar a transmissão.
    if width > max_width:
        new_height = int(height * max_width / width)
        frame = cv2.resize(
            frame,
            (max_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    # Converte o frame para JPEG.
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 70],
    )

    if not success:
        return

    # Armazena o frame e atualiza seu identificador de sequência.
    with jobs_lock:
        job = jobs.get(job_id)

        if job and job["status"] == "processing":
            job["live_frame"] = encoded.tobytes()
            job["frame_seq"] = job.get("frame_seq", 0) + 1


# Executa o processamento do vídeo e gerencia o estado da tarefa.
def process_job(job_id: str, input_path: Path) -> None:
    try:
        with jobs_lock:
            job = jobs.get(job_id)

            if job is None:
                return

            job.update(
                status="processing",
                stage="optimizing",
            )

        # Impede o uso simultâneo do modelo YOLO.
        with model_lock:
            model = get_model()

            # Executa o processamento e atualiza os dados pelos callbacks.
            output_path = process_video(
                input_path=input_path,
                model=model,
                progress_callback=lambda current, total: (
                    update_progress(job_id, current, total)
                ),
                frame_callback=lambda frame: (
                    publish_frame(job_id, frame)
                ),
                stage_callback=lambda stage: (
                    update_stage(job_id, stage)
                ),
                tracking_callback=lambda boxes, summary: (
                    update_tracking(job_id, boxes, summary)
                ),
            )

        # Registra a conclusão do processamento.
        with jobs_lock:
            job = jobs.get(job_id)

            if job is not None:
                job.update(
                    status="completed",
                    stage="completed",
                    progress=100,
                    output_path=str(output_path),
                    error=None,
                )

    # Registra possíveis erros durante o processamento.
    except Exception as exc:
        logger.exception(
            "Falha ao processar tarefa %s",
            job_id,
        )

        with jobs_lock:
            job = jobs.get(job_id)

            if job is not None:
                job.update(
                    status="failed",
                    stage="failed",
                    error=str(exc),
                )

    finally:
        # Remove o arquivo original após o processamento.
        input_path.unlink(missing_ok=True)


# Verifica se a API está em funcionamento.
@app.get("/health")
def health():
    return {"status": "ok"}


# Recebe um vídeo e inicia seu processamento em segundo plano.
@app.post("/videos", status_code=202)
async def upload_video(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    # Valida o formato do vídeo enviado.
    if not filename or extension not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Formato não suportado. "
                "Envie MP4, AVI, MOV, MKV ou WEBM."
            ),
        )

    # Gera um identificador único para o vídeo enviado.
    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{extension}"

    try:
        # Salva o vídeo em partes para evitar carregá-lo inteiro na memória.
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                destination.write(chunk)

    except Exception as exc:
        input_path.unlink(missing_ok=True)
        logger.exception("Falha ao salvar o vídeo enviado")

        raise HTTPException(
            status_code=500,
            detail="Não foi possível salvar o vídeo enviado.",
        ) from exc

    finally:
        await file.close()

    # Impede o processamento de arquivos vazios.
    if input_path.stat().st_size == 0:
        input_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="O arquivo está vazio.",
        )

    # Inicializa os dados de acompanhamento da tarefa.
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "frames_processed": 0,
            "total_frames": 0,
            "output_path": None,
            "error": None,
            "live_frame": None,
            "frame_seq": 0,
            "tracked_boxes": [],
            "tracking_summary": empty_tracking_summary(),
        }

    try:
        # Adiciona o vídeo à fila de processamento.
        executor.submit(process_job, job_id, input_path)

    except Exception as exc:
        input_path.unlink(missing_ok=True)

        with jobs_lock:
            jobs.pop(job_id, None)

        raise HTTPException(
            status_code=503,
            detail="Não foi possível iniciar o processamento.",
        ) from exc

    # Retorna o identificador e as rotas de acompanhamento.
    return {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress_url": f"/jobs/{job_id}",
        "download_url": f"/jobs/{job_id}/download",
    }


# Retorna o estado, progresso e resultados de uma tarefa.
@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Tarefa não encontrada.",
            )

        completed = job["status"] == "completed"

        # Remove informações internas da resposta.
        response = {
            key: value
            for key, value in job.items()
            if key not in {"output_path", "live_frame"}
        }

        # Disponibiliza as rotas do vídeo quando o processamento termina.
        response["video_url"] = (
            f"/jobs/{job_id}/video"
            if completed
            else None
        )

        response["download_url"] = (
            f"/jobs/{job_id}/download"
            if completed
            else None
        )

        return response


# Transmite os frames processados em tempo real via WebSocket.
@app.websocket("/ws/jobs/{job_id}")
async def stream_job_frames(
    websocket: WebSocket,
    job_id: str,
):
    await websocket.accept()
    last_seq = -1

    try:
        while True:
            with jobs_lock:
                job = jobs.get(job_id)

                if job is None:
                    frame = None
                    seq = -1
                    status = "missing"
                else:
                    frame = job.get("live_frame")
                    seq = job.get("frame_seq", 0)
                    status = job["status"]

            # Encerra a conexão caso a tarefa não exista.
            if status == "missing":
                await websocket.close(code=4404)
                return

            # Envia apenas frames que ainda não foram transmitidos.
            if frame is not None and seq != last_seq:
                await websocket.send_bytes(frame)
                last_seq = seq

            # Finaliza a transmissão após a conclusão ou falha da tarefa.
            if status in {"completed", "failed"}:
                break

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        pass

    except Exception:
        logger.exception(
            "Erro na transmissão WebSocket da tarefa %s",
            job_id,
        )

    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# Valida a tarefa e localiza o arquivo processado.
def get_job_output(job_id: str) -> tuple[Path, str]:
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Tarefa não encontrada.",
            )

        if job["status"] != "completed" or not job["output_path"]:
            raise HTTPException(
                status_code=409,
                detail="O vídeo ainda não está pronto.",
            )

        output_path = Path(job["output_path"])

        if not output_path.is_file():
            raise HTTPException(
                status_code=410,
                detail="O vídeo processado não está mais disponível.",
            )

        filename = f"{Path(job['filename']).stem}_detectado.mp4"

    return output_path, filename


# Disponibiliza o download do vídeo processado.
@app.get("/jobs/{job_id}/download")
def download_video(job_id: str):
    output_path, filename = get_job_output(job_id)

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=filename,
    )


# Permite visualizar o vídeo processado diretamente na aplicação.
@app.get("/jobs/{job_id}/video")
def get_processed_video(job_id: str):
    output_path, filename = get_job_output(job_id)

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=filename,
        content_disposition_type="inline",
    )


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
    input_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"

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
        with model_lock:
            model = get_model()

            output_path = await run_in_threadpool(
                process_image,
                input_path,
                model,
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


# Encerra o executor quando a aplicação é desligada.
@app.on_event("shutdown")
def shutdown_executor():
    executor.shutdown(wait=False, cancel_futures=True)