@echo off
echo ========================================================
echo Configurando el entorno de mmaction2...
echo ========================================================

REM Configurar rutas de Conda
set CONDA_PATH=C:\ProgramData\miniconda3
set CONDA_BAT=%CONDA_PATH%\condabin\conda.bat
set ACTIVATE_BAT=%CONDA_PATH%\Scripts\activate.bat

if not exist "%CONDA_BAT%" (
    echo [ERROR] No se encontro conda en %CONDA_PATH%
    echo Por favor, edite este archivo .bat con la ruta correcta a su Miniconda/Anaconda.
    pause
    exit /b 1
)

echo Aceptando Terminos de Servicio de Conda (evita errores)...
call "%CONDA_BAT%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>&1
call "%CONDA_BAT%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>&1
call "%CONDA_BAT%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>&1

echo Creando el entorno conda "mmaction2" en tu usuario si no existe...
call "%CONDA_BAT%" create -n mmaction2 python=3.9 -y

echo Activando el entorno mmaction2...
call "%ACTIVATE_BAT%" mmaction2

echo Instalando PyTorch (version CPU)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo Instalando OpenMIM...
python -m pip install -U openmim

echo Instalando dependencias de OpenMMLab (MMengine y MMCV)...
call mim install mmengine
call mim install "mmcv>=2.0.0rc4"

echo Instalando dependencias locales y registrando mmaction2...
python -m pip install -r requirements.txt
python -m pip install -v -e .

echo ========================================================
echo Configuracion completada con exito!
echo Para ejecutar el script de deteccion:
echo python detectar_evento.py D:\archive\dataset-video-split\test\Robbery030_x264.mp4 --checkpoint work_dirs/tsn_hurto/best_acc_top1_epoch_13.pth
echo ========================================================
pause
