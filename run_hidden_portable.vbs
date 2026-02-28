Option Explicit
Dim fso, sh, base, app
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("Wscript.Shell")
base = fso.GetParentFolderName(WScript.ScriptFullName)
app  = base & "\run_app_portable.cmd"
sh.Run """" & app & """", 0, False
