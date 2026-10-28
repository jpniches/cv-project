# Serviço responsável pelo processamento de vídeos utilizando YOLO.
# Realiza otimização, detecção, rastreamento e contagem de objetos.
# O processamento é acionado por uma rota definida no api.py.

from collections.abc import Callable
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.utils.model import get_model
from src.utils.video_optimizer import VideoOptimizer


# Diretório destinado ao armazenamento dos vídeos processados.
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs" / "predictions"


# Parâmetros utilizados na detecção e no rastreamento dos objetos.
CONFIDENCE = 0.25
IMAGE_SIZE = 640
MIN_TRACK_SECONDS = 1.5
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# Classes de objetos utilizadas no treinamento do modelo.
EXPECTED_CLASSES = (
    "caixa-preta",
    "caixa-verde",
    "caixa-azul",
    "caixa-cinza",
    "caixa-vermelha",
    "caixa-fechada",
)


# Tipos de callbacks utilizados para acompanhar o processamento.
ProgressCallback = Callable[[int, int], None]
FrameCallback = Callable[[object], None]
StageCallback = Callable[[str], None]
TrackingCallback = Callable[[list[dict], dict], None]


# Inicializa o resumo de rastreamento com os contadores zerados.
def empty_tracking_summary() -> dict:
    return {
        "total": 0,
        **{class_name: 0 for class_name in EXPECTED_CLASSES},
        "classes_nao_reconhecidas": {},
    }


# Contabiliza os objetos rastreados por classe, incluindo classes desconhecidas.
def build_tracking_summary(boxes: list[dict]) -> dict:

    summary = empty_tracking_summary()
    summary["total"] = len(boxes)

    for item in boxes:
        class_name = item["classe"]

        if class_name in EXPECTED_CLASSES:
            summary[class_name] += 1
        else:
            unknown = summary["classes_nao_reconhecidas"]
            unknown[class_name] = unknown.get(class_name, 0) + 1

    return summary


# Define o caminho de saída e evita sobrescrever vídeos já existentes.
def get_output_path(input_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{input_path.stem}_detectado.mp4"
    counter = 1

    while output_path.exists():
        output_path = OUTPUT_DIR / (
            f"{input_path.stem}_detectado_{counter}.mp4"
        )
        counter += 1

    return output_path


# Executa o processamento completo do vídeo, desde a otimização
# até a geração do arquivo com as detecções e rastreamentos.
def process_video(
    input_path: str | Path,
    model: YOLO | None = None,
    progress_callback: ProgressCallback | None = None,
    frame_callback: FrameCallback | None = None,
    stage_callback: StageCallback | None = None,
    tracking_callback: TrackingCallback | None = None,
) -> Path:

    input_path = Path(input_path).resolve()

    # Valida a existência do arquivo e seu formato antes do processamento.
    if not input_path.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {input_path}")

    if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Formato de vídeo não suportado: {input_path.suffix}"
        )

    # Carrega o modelo de detecção e inicializa o otimizador de vídeos.
    model = model or get_model()
    optimizer = VideoOptimizer()

    optimized_path = None
    capture = None
    writer = None
    output_path = None

    frame_count = 0
    total_frames = 0

    # Armazena os objetos identificados e os rastreamentos em validação.
    tracked_objects: dict[int, dict] = {}
    pending_tracks: dict[int, dict] = {}
    next_track_order = 1

    # Envia periodicamente os objetos rastreados e o resumo atualizado.
    def publish_tracking(force: bool = False) -> None:
        if tracking_callback is None:
            return

        # Publica a cada 10 frames para evitar atualizações desnecessárias.
        if not force and frame_count % 10 != 0:
            return

        boxes = sorted(
            (item.copy() for item in tracked_objects.values()),
            key=lambda item: item["ordem"],
        )

        tracking_callback(
            boxes,
            build_tracking_summary(boxes),
        )

    try:
        if stage_callback:
            stage_callback("optimizing")

        # Otimiza o vídeo de entrada antes de iniciar a detecção.
        print(f"Otimizando vídeo: {input_path.name}")
        optimized_path = optimizer.optimize(input_path)

        capture = cv2.VideoCapture(str(optimized_path))

        if not capture.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir o vídeo otimizado: {optimized_path}"
            )

        # Obtém as propriedades do vídeo necessárias para o processamento.
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise RuntimeError(
                "O vídeo otimizado possui dimensões inválidas."
            )

        # Define um FPS padrão caso os metadados sejam inválidos.
        if fps <= 0:
            fps = 30.0

        # Converte o tempo mínimo de rastreamento em quantidade de frames.
        min_track_frames = max(
            1,
            round(fps * MIN_TRACK_SECONDS),
        )

        output_path = get_output_path(input_path)

        # Inicializa a gravação do vídeo com as detecções identificadas.
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Não foi possível criar o vídeo de saída."
            )

        if stage_callback:
            stage_callback("detecting")

        if progress_callback:
            progress_callback(0, total_frames)

        print(
            f"Processando {input_path.name}: "
            f"{width}x{height}, {fps:.2f} FPS, "
            f"{total_frames} frames estimados"
        )

        # Percorre os frames do vídeo até finalizar sua leitura.
        while True:
            success, frame = capture.read()

            if not success:
                break

            # Detecta e rastreia os objetos utilizando YOLO e ByteTrack.
            results = model.track(
                source=frame,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )

            result = results[0]
            annotated_frame = result.plot()
            frame_count += 1

            boxes = result.boxes

            # Extrai os identificadores, classes e níveis de confiança.
            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for track_id, class_id, confidence in zip(
                    track_ids,
                    class_ids,
                    confidences,
                ):
                    class_name = result.names.get(
                        class_id,
                        str(class_id),
                    )

                    # Verifica se o objeto já possui um registro de rastreamento.
                    pending = pending_tracks.get(track_id)

                    # Registra novos objetos ou atualiza os já identificados.
                    if pending is None:
                        pending = {
                            "id": track_id,
                            "ordem": next_track_order,
                            "classe": class_name,
                            "confianca": round(float(confidence), 4),
                            "primeiro_frame": frame_count,
                            "ultimo_frame": frame_count,
                            "contagem_deteccoes": 1,
                        }

                        pending_tracks[track_id] = pending
                        next_track_order += 1
                    else:
                        pending["classe"] = class_name
                        pending["confianca"] = round(float(confidence), 4)
                        pending["ultimo_frame"] = frame_count
                        pending["contagem_deteccoes"] += 1

                    # Calcula o intervalo de frames em que o objeto foi rastreado.
                    duration_frames = (
                        pending["ultimo_frame"]
                        - pending["primeiro_frame"]
                        + 1
                    )

                    # Considera apenas objetos rastreados pelo tempo mínimo.
                    if duration_frames >= min_track_frames:
                        tracked_objects[track_id] = pending

            # Grava o frame com as detecções no vídeo de saída.
            writer.write(annotated_frame)

            # Atualiza a visualização do frame, quando disponível.
            if frame_callback:
                frame_callback(annotated_frame)

            # Atualiza o progresso do processamento.
            if progress_callback:
                progress_callback(frame_count, total_frames)

            # Publica periodicamente os dados dos objetos rastreados.
            publish_tracking()

            if frame_count % 100 == 0:
                print(f"Frames processados: {frame_count}")

        if frame_count == 0:
            raise RuntimeError(
                "Nenhum frame foi lido do vídeo otimizado."
            )

        # Publica o resumo final após o processamento de todos os frames.
        publish_tracking(force=True)

    # Remove arquivos de saída incompletos caso ocorra algum erro.
    except Exception:
        if output_path is not None and output_path.exists():
            output_path.unlink(missing_ok=True)
        raise

    # Libera os recursos utilizados e limpa o estado do rastreador.
    finally:
        if capture is not None:
            capture.release()

        if writer is not None:
            writer.release()

        # Reinicia o rastreador para evitar interferências entre vídeos.
        predictor = getattr(model, "predictor", None)

        if predictor is not None:
            for tracker in getattr(predictor, "trackers", []):
                tracker.reset()

        # Remove o arquivo temporário gerado durante a otimização.
        if optimized_path is not None:
            optimizer.cleanup(optimized_path)

    # Verifica se o vídeo processado foi gerado corretamente.
    if (
        output_path is None
        or not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "O vídeo processado não foi gerado corretamente."
        )

    # Atualiza o progresso e informa a conclusão do processamento.
    if progress_callback:
        progress_callback(frame_count, total_frames or frame_count)

    if stage_callback:
        stage_callback("completed")

    # Exibe as informações finais do processamento.
    print(f"Frames processados: {frame_count}")
    print(f"Objetos rastreados: {len(tracked_objects)}")
    print(f"Resumo: {build_tracking_summary(list(tracked_objects.values()))}")
    print(f"Resultado salvo em: {output_path}")

    # Retorna o caminho do vídeo processado para utilização pela aplicação.
    return output_path