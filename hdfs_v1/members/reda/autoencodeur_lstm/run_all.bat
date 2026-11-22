@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if not exist logs mkdir logs
if exist logs\ECHEC.txt del logs\ECHEC.txt
if exist logs\TERMINE.txt del logs\TERMINE.txt

echo ============================================================
echo  LogSentinel - autoencodeur LSTM - lancement complet
echo  Laisser le PC branche, couvercle ouvert. Ne pas fermer cette fenetre.
echo ============================================================

set PY=
where py >nul 2>nul && set PY=py -3
if "%PY%"=="" set PY=python
%PY% --version
if errorlevel 1 (
  echo Python introuvable. Installer Python 3.12 depuis python.org en cochant "Add python.exe to PATH".
  pause
  exit /b 1
)

if not exist .venv\Scripts\python.exe (
  echo [0/3] Creation de l'environnement Python...
  %PY% -m venv .venv
  if errorlevel 1 goto erreur
)
set VPY=.venv\Scripts\python.exe

echo [0/3] Installation des bibliotheques (premiere fois : quelques minutes)...
%VPY% -m pip install --upgrade pip >> logs\0_installation.log 2>&1
%VPY% -c "import torch" 2>nul
if errorlevel 1 (
  %VPY% -m pip install torch --index-url https://download.pytorch.org/whl/cpu >> logs\0_installation.log 2>&1
  if errorlevel 1 goto erreur
)
%VPY% -m pip install -r requirements.txt >> logs\0_installation.log 2>&1
if errorlevel 1 goto erreur
%VPY% -c "import torch, pandas, pyarrow, sklearn, matplotlib; print('bibliotheques OK, torch', torch.__version__)"
if errorlevel 1 goto erreur

echo [1/3] Preparation des donnees...
%VPY% src\prepare_data.py
if errorlevel 1 goto erreur

echo [2/3] Comparaison des reglages sur val...
%VPY% src\train_tune.py
if errorlevel 1 goto erreur

echo [3/3] Entrainement final et test...
%VPY% src\final_test.py
if errorlevel 1 goto erreur

echo.
echo ============================================================
echo  TERMINE. Resultats : results\rapport_nuit.md
echo ============================================================
echo TERMINE > logs\TERMINE.txt
pause
exit /b 0

:erreur
echo.
echo ============================================================
echo  Une etape a echoue. Voir le dossier logs (dernier fichier .log).
echo  Relancer run_all.bat reprend la ou ca s'est arrete.
echo ============================================================
echo ECHEC > logs\ECHEC.txt
pause
exit /b 1
