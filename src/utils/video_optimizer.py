# Arquivo usado para otimizar os vídeos, para aparelhos com qualidade maior 
# que os seus vídeos pesam muito e demoram para fazer upload, feito com ffmpeg
import subprocess
import tempfile
from pathlib import Path

class VideoOptimizer:
    def __init__(
        self,
        output_width: int = 1280,
        output_height: int = 720,
        fps: int = 15,
        video_codec: str = "libx264",
        preset: str = "ultrafast",
        crf: int = 28,
    ):
        # Define os parâmetros de resolução, FPS e compressão do vídeo.
        self.output_width = output_width
        self.output_height = output_height
        self.fps = fps
        self.video_codec = video_codec
        self.preset = preset
        self.crf = crf

    def optimize(self, input_path: str | Path) -> Path:
        input_path = Path(input_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(
                f"Vídeo não encontrado: {input_path}"
            )

        if not input_path.is_file():
            raise ValueError(
                f"O caminho não é um arquivo: {input_path}"
            )

        # Cria um diretório temporário para armazenar o vídeo otimizado.
        output_dir = Path(tempfile.mkdtemp(prefix="cv_video_"))
        output_path = output_dir / "optimized.mp4"

        # Redimensiona o vídeo preservando a proporção e ajustando o formato.
        scale_filter = (
            f"scale={self.output_width}:{self.output_height}:"
            "force_original_aspect_ratio=decrease,"
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,"
            "setsar=1"
        )

        # Configura o ffmpeg para gerar um vídeo compatível com o processamento.
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(input_path),
            "-map", "0:v:0",
            "-an",
            "-vf", scale_filter,
            "-r", str(self.fps),
            "-c:v", self.video_codec,
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            "-threads", "1",
            "-movflags", "+faststart",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    "Falha ao otimizar o vídeo: "
                    f"{result.stderr.strip()}"
                )

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(
                    "O ffmpeg não gerou um vídeo válido."
                )

            return output_path

        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg não encontrado. Verifique a instalação "
                "e a variável PATH do Windows."
            ) from exc

    @staticmethod
    def cleanup(optimized_path: str | Path) -> None:
        optimized_path = Path(optimized_path)

        try:
            # Remove o vídeo temporário e seu diretório, se estiver vazio.
            optimized_path.unlink(missing_ok=True)
            optimized_path.parent.rmdir()
        except OSError:
            pass