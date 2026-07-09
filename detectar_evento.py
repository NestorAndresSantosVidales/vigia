import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

LABEL_FILE = Path("D:/archive/dataset-video-split/lable.txt")
DEFAULT_CONFIG = "configs/recognition/tsn/tsn_hurto.py"
DEFAULT_CHECKPOINT = "work_dirs/tsn_hurto/best_acc_top1_epoch_8.pth"

THEFT_CLASSES = {"Stealing", "Shoplifting", "Robbery", "Burglary"}
RISK_CLASSES = {"Fighting", "Assault", "Abuse", "Vandalism", "Shooting", "Explosion", "Arson"}


def load_labels(label_file):
    labels = {}

    for line in label_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        parts = line.split()
        label_id = int(parts[-1])
        label_name = " ".join(parts[:-1])
        labels[label_id] = label_name

    return labels


def run_inference(video, config, checkpoint, device):
    cmd = [
        sys.executable,
        "demo/demo_inferencer.py",
        str(video),
        "--rec", config,
        "--rec-weights", checkpoint,
        "--device", device,
        "--print-result",
    ]

    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.returncode != 0:
        print("\nNo se pudo ejecutar la inferencia.")
        print("\nDetalle tecnico:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    prediction_line = None

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{'predictions':"):
            prediction_line = line

    if prediction_line is None:
        print("\nNo se pudo leer la prediccion del modelo.")
        print("\nSalida recibida:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    return ast.literal_eval(prediction_line)


def explain(prediction, labels):
    scores = prediction["predictions"][0]["rec_scores"][0]
    top = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:5]

    top_id, top_score = top[0]
    top_label = labels.get(top_id, f"Clase {top_id}")

    theft_score = sum(
        score for idx, score in enumerate(scores)
        if labels.get(idx) in THEFT_CLASSES
    )

    risk_score = sum(
        score for idx, score in enumerate(scores)
        if labels.get(idx) in RISK_CLASSES
    )

    normal_score = scores[6] if len(scores) > 6 else 0

    print("\n==============================")
    print(" Resultado del analisis")
    print("==============================")
    print(f"Evento mas probable: {top_label}")
    print(f"Confianza del modelo: {top_score * 100:.1f}%")

    print("\nTop 5 posibilidades:")
    for idx, score in top:
        label = labels.get(idx, f"Clase {idx}")
        print(f" - {label}: {score * 100:.1f}%")

    print("\nLectura para una persona no tecnica:")

    if theft_score >= 0.50:
        print("ALERTA ALTA: el video tiene senales fuertes de posible hurto.")
    elif theft_score >= 0.25:
        print("ALERTA MEDIA: el video podria estar relacionado con hurto, requiere revision humana.")
    elif risk_score >= 0.40:
        print("ALERTA DE RIESGO: no parece hurto directo, pero se ve comportamiento sospechoso o violento.")
    elif normal_score >= 0.35:
        print("SIN ALERTA PRINCIPAL: el modelo lo interpreta principalmente como actividad normal.")
    else:
        print("RESULTADO INCIERTO: el modelo no tiene suficiente seguridad. Requiere revision humana.")

    print("\nResumen de riesgo:")
    print(f" - Probabilidad agrupada de hurto: {theft_score * 100:.1f}%")
    print(f" - Probabilidad agrupada de riesgo/sospecha: {risk_score * 100:.1f}%")
    print(f" - Probabilidad de normalidad: {normal_score * 100:.1f}%")

    print("\nNota:")
    print("Este resultado no confirma un delito. Solo genera una alerta visual para revision humana.")


def main():
    parser = argparse.ArgumentParser(
        description="Analisis interactivo de video con MMAction2"
    )
    parser.add_argument("video", help="Ruta del video MP4 a analizar")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    video = Path(args.video)

    if not video.exists():
        print(f"No encontre el video: {video}")
        sys.exit(1)

    if not LABEL_FILE.exists():
        print(f"No encontre el archivo de etiquetas: {LABEL_FILE}")
        sys.exit(1)

    if not Path(args.config).exists():
        print(f"No encontre el config: {args.config}")
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        print(f"No encontre el checkpoint: {args.checkpoint}")
        sys.exit(1)

    print("\nAnalizando video...")
    print(f"Archivo: {video}")
    print("Esto puede tardar unos segundos.\n")

    labels = load_labels(LABEL_FILE)
    prediction = run_inference(video, args.config, args.checkpoint, args.device)
    explain(prediction, labels)


if __name__ == "__main__":
    main()