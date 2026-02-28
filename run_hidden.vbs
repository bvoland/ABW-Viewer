' run_hidden.vbs — startet Streamlit ohne Fenster
Option Explicit
Dim fso, sh, base, pyw, cmd, app

Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("Wscript.Shell")
base = fso.GetParentFolderName(WScript.ScriptFullName)
pyw  = base & "\.venv\Scripts\pythonw.exe"

' Falls die venv noch nicht existiert: einmal Setup-CMD im Hintergrund starten
If Not fso.FileExists(pyw) Then
  app = base & "\run_app.cmd"
  sh.Run """" & app & """", 0, False
  WScript.Quit
End If

' Streamlit ohne Konsole starten
cmd = """" & pyw & """ -m streamlit run """ & base & "\ABW_Datenvisualisierung.py"" --server.address=localhost --server.port=8501"
sh.Run cmd, 0, False
