@echo off
setlocal
rem ==== ins Skriptverzeichnis wechseln (Leerzeichen-tauglich) ====
cd /d "%~dp0"

rem ==== portable Python Basis ====
set "PORTABLE=%cd%\portable_python"

rem ==== python.exe unter portable_python rekursiv suchen ====
set "PY="
for /f "delims=" %%F in ('dir /b /s "%PORTABLE%\python.exe" 2^>nul') do (
  set "PY=%%F"
  goto :found_py
)
echo [ERROR] Konnte portable_python\...\python.exe nicht finden.
echo Bitte WinPython 3.12 (Zero) ins Verzeichnis portable_python\ entpacken.
pause
exit /b 1

:found_py
rem PY z.B.: ...\python-3.12.x.amd64\python.exe
set "PYDIR=%PY:python.exe=%"
set "PYW=%PYDIR%pythonw.exe"
set "PIP=%PYDIR%Scripts\pip.exe"

if not exist "%PYW%" (
  echo [ERROR] pythonw.exe nicht gefunden: %PYW%
  echo Pruefe deine WinPython-Entpackung.
  pause
  exit /b 1
)

rem ==== Prüfen, ob streamlit bereits installiert ist ====
"%PY%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
  rem ---- Offline-Install aus wheelhouse ----
  if not exist "wheelhouse" (
    echo [ERROR] wheelhouse\ fehlt. Bitte Wheels mit Python 3.12 vorab herunterladen.
    pause
    exit /b 1
  )

  rem Falls pip nicht da ist, sicherheitshalber bootstrappen
  "%PY%" -m pip --version >nul 2>&1
  if errorlevel 1 (
    echo [INFO] pip fehlt – versuche ensurepip...
    "%PY%" -m ensurepip --upgrade
  )

  echo [INFO] Installiere Pakete offline aus wheelhouse (cp312/win_amd64)...
  if exist requirements.txt (
    "%PIP%" install --no-index --find-links "%cd%\wheelhouse" -r requirements.txt
  ) else (
    "%PIP%" install --no-index --find-links "%cd%\wheelhouse" ^
      streamlit pandas numpy plotly sqlalchemy psycopg2-binary duckdb
  )
)

rem ==== App starten (ohne Konsole via pythonw.exe) ====
start "" "%PYW%" -m streamlit run "ABW_Datenvisualisierung.py" --server.address=localhost --server.port=8501
exit /b 0
