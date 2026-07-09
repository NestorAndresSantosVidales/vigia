import streamlit as st
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

st.title("🎥 MMAction2 - Detector de Eventos de Seguridad")
st.write("Sube un fragmento de video para clasificar la actividad detectada en tiempo real.")

uploaded_file = st.file_uploader("Selecciona un archivo de video MP4", type=["mp4"])

if uploaded_file is not None:
    # Guardar video temporalmente
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_video_path = temp_dir / uploaded_file.name
    
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
        
    st.video(str(temp_video_path))
    
    if st.button("Analizar Video"):
        with st.spinner("Analizando video... Esto puede tardar unos segundos."):
            labels = load_labels(DEFAULT_LABELS)
            prediction, error = run_inference(temp_video_path)
            
            if error:
                st.error(f"Error al procesar el video:\n{error}")
            else:
                scores = prediction["predictions"][0]["rec_scores"][0]
                top = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:5]
                
                theft_score = sum(scores[idx] for idx, name in labels.items() if name in THEFT_CLASSES)
                risk_score = sum(scores[idx] for idx, name in labels.items() if name in RISK_CLASSES)
                normal_score = scores[6] if 6 in labels else 0.0
                
                top_id, top_score = top[0]
                top_label = labels.get(top_id, f"Clase {top_id}")
                
                st.subheader("📊 Resultado del Análisis")
                st.write(f"**Evento más probable:** `{top_label}` ({top_score * 100:.1f}% de confianza)")
                
                st.markdown("#### 🧑‍💼 Interpretación para Lectura Humana:")
                if theft_score >= 0.50:
                    st.error("🚨 **ALERTA ALTA:** El video tiene señales muy fuertes de posible **hurto**.")
                elif theft_score >= 0.25:
                    st.warning("⚠️ **ALERTA MEDIA:** El video podría estar relacionado con **hurto** (requiere revisión humana).")
                elif risk_score >= 0.40:
                    st.info("🔥 **ALERTA DE RIESGO:** No parece hurto directo, pero se observa **comportamiento sospechoso, violento o de riesgo**.")
                elif normal_score >= 0.35:
                    st.success("✅ **SIN ALERTA PRINCIPAL:** Actividad interpretada principalmente como **normal**.")
                else:
                    st.write("❓ **RESULTADO INCIERTO:** El modelo no tiene suficiente certeza. Requiere revisión humana.")
                    
                st.markdown("#### 📈 Resumen de Riesgos Agrupados:")
                st.write(f"- **Probabilidad de Hurto:** {theft_score * 100:.1f}%")
                st.write(f"- **Probabilidad de Sospecha / Riesgo:** {risk_score * 100:.1f}%")
                st.write(f"- **Probabilidad de Actividad Normal:** {normal_score * 100:.1f}%")
                
                st.subheader("📋 Top 5 Posibilidades Detalladas")
                for idx, score in top:
                    label = labels.get(idx, f"Clase {idx}")
                    st.progress(float(score), text=f"{label}: {score * 100:.1f}%")
                    
    # Limpieza
    if temp_video_path.exists():
        try:
            os.remove(temp_video_path)
        except Exception:
            pass
