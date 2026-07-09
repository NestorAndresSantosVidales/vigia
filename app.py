import gradio as gr
import os
import sys
import ast
import subprocess
from pathlib import Path

# Configuración básica
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = str(PROJECT_ROOT / "configs/recognition/tsn/tsn_hurto.py")
DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "work_dirs/tsn_hurto/best_acc_top1_epoch_13.pth")
DEFAULT_LABELS = str(PROJECT_ROOT / "lable.txt")

THEFT_CLASSES = {"Stealing", "Shoplifting", "Robbery", "Burglary"}
RISK_CLASSES = {"Fighting", "Assault", "Abuse", "Vandalism", "Shooting", "Explosion", "Arson"}

def load_labels(label_file_path):
    labels = {}
    with open(label_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            label_id = int(parts[-1])
            label_name = " ".join(parts[:-1])
            labels[label_id] = label_name
    return labels

def run_inference(video_path):
    # Ejecuta demo/demo_inferencer.py de la misma manera que detectar_evento.py
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "demo/demo_inferencer.py"),
        str(video_path),
        "--rec", DEFAULT_CONFIG,
        "--rec-weights", DEFAULT_CHECKPOINT,
        "--device", "cpu",
        "--print-result",
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    
    if result.returncode != 0:
        return None, f"Error en inferencia:\n{result.stdout}\n{result.stderr}"
        
    prediction_line = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{'predictions':"):
            prediction_line = line
            
    if prediction_line is None:
        return None, f"No se pudo leer la predicción del modelo. Salida:\n{result.stdout}\n{result.stderr}"
        
    return ast.literal_eval(prediction_line), None

def analyze_video(video):
    if not video:
        return "Por favor, suba un video.", {}
        
    if not os.path.exists(DEFAULT_LABELS):
        return f"Error: No se encontró el archivo de etiquetas en {DEFAULT_LABELS}", {}
        
    labels = load_labels(DEFAULT_LABELS)
    prediction, error = run_inference(video)
    
    if error:
        return f"Error al procesar el video:\n{error}", {}
        
    scores = prediction["predictions"][0]["rec_scores"][0]
    top = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:5]
    
    # Calcular probabilidades agregadas
    theft_score = sum(scores[idx] for idx, name in labels.items() if name in THEFT_CLASSES)
    risk_score = sum(scores[idx] for idx, name in labels.items() if name in RISK_CLASSES)
    normal_score = scores[6] if 6 in labels else 0.0
    
    # Evento principal
    top_id, top_score = top[0]
    top_label = labels.get(top_id, f"Clase {top_id}")
    
    # Formatear la explicación en Markdown
    result_text = f"### 📊 Resultado del Análisis\n"
    result_text += f"**Evento más probable:** `{top_label}` ({top_score * 100:.1f}% de confianza)\n\n"
    
    result_text += "#### 🧑‍💼 Interpretación para Lectura Humana:\n"
    if theft_score >= 0.50:
        result_text += "🚨 **ALERTA ALTA:** El video tiene señales muy fuertes de posible **hurto**.\n"
    elif theft_score >= 0.25:
        result_text += "⚠️ **ALERTA MEDIA:** El video podría estar relacionado con **hurto** (requiere revisión humana).\n"
    elif risk_score >= 0.40:
        result_text += "🔥 **ALERTA DE RIESGO:** No parece hurto directo, pero se observa **comportamiento sospechoso, violento o de riesgo**.\n"
    elif normal_score >= 0.35:
        result_text += "✅ **SIN ALERTA PRINCIPAL:** Actividad interpretada principalmente como **normal**.\n"
    else:
        result_text += "❓ **RESULTADO INCIERTO:** El modelo no tiene suficiente certeza. Requiere revisión humana.\n"
        
    result_text += f"\n#### 📈 Resumen de Riesgos Agrupados:\n"
    result_text += f"- **Probabilidad de Hurto:** {theft_score * 100:.1f}%\n"
    result_text += f"- **Probabilidad de Sospecha / Riesgo:** {risk_score * 100:.1f}%\n"
    result_text += f"- **Probabilidad de Actividad Normal:** {normal_score * 100:.1f}%\n"
    
    result_text += "\n*Nota: Este resultado es generado por un modelo de Inteligencia Artificial para asistencia visual y no constituye una confirmación legal de ningún delito.*"
    
    # Formatear para el gráfico de etiquetas de Gradio
    chart_data = {labels.get(idx, f"Clase {idx}"): float(score) for idx, score in top}
    
    return result_text, chart_data

# Construir interfaz Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎥 MMAction2 - Detector de Eventos de Seguridad
        Sube un fragmento de video para clasificar la actividad detectada en tiempo real.
        """
    )
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Subir Video (MP4)")
            submit_btn = gr.Button("Analizar Video", variant="primary")
            
        with gr.Column():
            output_markdown = gr.Markdown(label="Interpretación")
            output_chart = gr.Label(label="Top 5 Probabilidades", num_top_classes=5)
            
    submit_btn.click(
        fn=analyze_video,
        inputs=[video_input],
        outputs=[output_markdown, output_chart]
    )

if __name__ == "__main__":
    demo.launch()
