@echo off
setlocal
rem ==== immer ins Skriptverzeichnis wechseln (wichtig bei OneDrive & Leerzeichen) ====
cd /d "%~dp0"

rem ==== Python-Launcher finden (py oder python) ====
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)

rem ==== Defekte venv erkennen und bei Bedarf neu erstellen ====
if exist ".venv\Scripts\python.exe" (
  call ".venv\Scripts\python.exe" -V >nul 2>&1
  if errorlevel 1 (
    echo [INFO] Bestehende .venv ist ungueltig und wird neu erstellt. >run_log.txt
    rmdir /s /q ".venv"
  )
)

rem ==== Virtuelle Umgebung anlegen, falls fehlt ====
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv ".venv"
  if errorlevel 1 (
    echo [ERROR] Konnte venv nicht erstellen. Ist Python installiert?
    pause
    exit /b 1
  )
)

rem ==== Pakete installieren/aktualisieren ====
call ".venv\Scripts\python.exe" -m pip install --upgrade pip >run_log.txt 2>&1
if exist requirements.txt (
  call ".venv\Scripts\python.exe" -m pip install -r requirements.txt >>run_log.txt 2>&1
) else (
  call ".venv\Scripts\python.exe" -m pip install streamlit pandas numpy plotly sqlalchemy psycopg2-binary duckdb >>run_log.txt 2>&1
)

rem ==== Streamlit starten (ohne streamlit.exe; sicher mit -m) ====
start "" ".venv\Scripts\python.exe" -m streamlit run "ABW_Datenvisualisierung.py" --server.address=localhost --server.port=8501
exit /b 0
