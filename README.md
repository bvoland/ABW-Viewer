# ABW Daten Visualisierung

Streamlit-App zur Analyse von Sensordaten aus PostgreSQL mit optionalem lokalem DuckDB-Cache.

Der aktuell genutzte Einstiegspunkt ist `ABW_Datenvisualisierung.py`. Historische Zwischenstände liegen im Ordner `legacy/` und werden von den Startskripten nicht verwendet.

## Zweck

Die App kombiniert:

- Metadaten aus `nodes_hierarchy`
- Zeitreihen aus `nodes_history`
- Umweltdaten aus `nodes_history_env`

und stellt diese gefiltert als Diagramme, Kennzahlen und CSV-Export bereit.

## Funktionen

- Filter nach `Building`, `Floor`, `Area`, `Sensor type`, `Vanity`, `Channel`
- frei wählbarer Datums- und Zeitbereich
- optionale Aggregation per Resampling
- optionale Glättung per Rolling Window
- Batch-Laden mit Fortschrittsanzeige
- lokaler DuckDB-Cache für wiederholte Abfragen
- CSV-Export der geladenen Daten
- einfacher Self-Test für DB- und Schema-Prüfung

## Projektstruktur

- `ABW_Datenvisualisierung.py`: aktive Streamlit-App
- `run_app.cmd`: Start mit lokaler `.venv`
- `run_hidden.vbs`: Start ohne sichtbare Konsole
- `run_app_portable.cmd`: Start mit portablem Python
- `run_hidden_portable.vbs`: versteckter Start des portablen Launchers
- `db_config.local.json.example`: Vorlage für lokale DB-Zugangsdaten
- `Dockerfile`: Container-Build für Server-Deployment
- `deploy/`: Reverse-Proxy- und Compose-Dateien für Serverbetrieb
- `requirements.txt`: Python-Abhängigkeiten
- `sensor_cache.duckdb`: lokaler Cache, wird zur Laufzeit befüllt
- `legacy/`: archivierte App-Zwischenstände

## Voraussetzungen

### Option 1: Standard-Start mit lokaler Python-Installation

- Windows
- Python 3.12 mit `py` oder `python` im `PATH`
- Netzwerkzugriff auf die PostgreSQL-Datenbank

Start:

```bat
run_app.cmd
```

Das Skript:

- wechselt ins Projektverzeichnis
- erstellt bei Bedarf eine `.venv`
- installiert Abhängigkeiten aus `requirements.txt`
- startet Streamlit auf `http://localhost:8501`

Wichtig: Eine defekte `.venv` wird automatisch neu erstellt. Das ist relevant, wenn die virtuelle Umgebung ursprünglich auf eine nicht mehr vorhandene Python-Installation zeigt.

### Option 2: Portabler Start ohne lokale Python-Installation

```bat
run_app_portable.cmd
```

Voraussetzungen:

- `portable_python\\...\\python.exe` muss vorhanden sein
- `wheelhouse\\` enthält die benötigten Wheels für Python 3.12 auf Windows

### Direkter Start im Terminal

```bat
streamlit run "ABW_Datenvisualisierung.py"
```

Nicht mit `python ABW_Datenvisualisierung.py` starten, da die Datei als Streamlit-App ausgelegt ist.

## Server-Deployment

Für den Serverbetrieb ist eine schlanke Docker-Struktur vorgesehen:

- Caddy als Reverse Proxy
- ein eigener App-Container für ABW
- gemeinsames Docker-Netzwerk `web` für weitere Apps

Relevante Dateien:

- `deploy/proxy/docker-compose.yml`
- `deploy/proxy/Caddyfile`
- `deploy/abw/docker-compose.yml`

Die produktive DB-Konfiguration wird nicht committed, sondern serverseitig als:

- `deploy/abw/data/db_config.local.json`

bereitgestellt.

## Konfiguration

Die Datenbankzugänge liegen nicht mehr im versionierten Code. Es gibt zwei unterstützte Wege:

- lokale Datei `db_config.local.json` auf Basis von `db_config.local.json.example`
- Umgebungsvariablen

Unterstützte Umgebungsvariablen:

- `ABW_DB_USER`
- `ABW_DB_PASS`
- `ABW_DB_HOST`
- `ABW_DB_PORT`
- `ABW_DB_NAME`

Umgebungsvariablen überschreiben Werte aus `db_config.local.json`.

## Datenmodell-Annahmen

Die App erwartet im Wesentlichen:

- `nodes_hierarchy(id, vanity, building, floor, area, nodeTypeName|Alias)`
- `nodes_history(timestamp, value, ID, type)`
- `nodes_history_env(timestamp, value, ID, type)`

Für `nodeTypeName` werden mehrere Alias-Namen toleriert, darunter:

- `nodeTypeName`
- `nodetypename`
- `node_type_name`
- `nodetype`
- `type`

## Cache-Verhalten

Wenn `Use local cache (DuckDB)` aktiviert ist, liest die App bevorzugt aus `sensor_cache.duckdb` und schreibt frisch geladene Daten dort hinein.

Wenn `Use cache only (no DB fetch)` aktiviert ist:

- werden nur lokal gecachte Daten verwendet
- findet kein Fallback auf PostgreSQL statt
- leere Ergebnisse bedeuten dann entweder fehlende Daten oder einen unvollständigen Cache

## Bekannte Besonderheiten

- Die App mischt derzeit deutsche Kommentare mit englischer UI. Das ist funktional unkritisch, aber dokumentationsrelevant.
- Historische Versionen sind nur noch als Referenz unter `legacy/` abgelegt.
- Die Datei `sensor_cache.duckdb` kann sehr groß werden und gehört nicht in ein Git-Repository.
- Lokale DB-Zugangsdaten gehören in `db_config.local.json` oder in Umgebungsvariablen, nicht ins Repository.

## Empfohlener Git-Umfang

Ins Repository sollten nur Quelltexte und Projektdateien aufgenommen werden. Nicht versioniert werden sollten insbesondere:

- `.venv/`
- `portable_python/`
- `wheelhouse/`
- `sensor_cache.duckdb`
- `run_log.txt`
- `db_config.local.json`
- `__pycache__/`

## Änderungsstand dieser Bereinigung

Im Rahmen der ersten Bereinigung wurden folgende Inkonsistenzen adressiert:

- `requirements.txt` ergänzt, damit beide Launcher konsistent installieren
- `.gitignore` ergänzt, damit keine riesigen Laufzeitartefakte eingecheckt werden
- `run_app.cmd` auf defekte `.venv` vorbereitet
- `run_app_portable.cmd` auf `requirements.txt` umgestellt
- `Use cache only` so korrigiert, dass kein unbeabsichtigter DB-Fallback mehr erfolgt
- Self-Test für `nodes_hierarchy` an die tatsächlich unterstützten Alias-Spalten angepasst
- DB-Credentials aus dem versionierten Code entfernt und auf lokale Konfiguration umgestellt
- historische Dateiversionen in `legacy/` archiviert
