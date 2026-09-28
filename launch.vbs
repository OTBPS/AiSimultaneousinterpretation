' Hidden launcher for the desktop shortcut.
'
' Why this exists rather than pointing the shortcut straight at pythonw.exe:
' Explorer's environment is not a shell's, and resolving pythonw.exe through
' PATH is the part most likely to differ. This pins the interpreter to the
' known install and only falls back to PATH.
'
' An earlier version of this file also piped both streams through
' "cmd /c ... > out 2> err". That was a mistake: the running instance keeps
' those files open, so a SECOND launch could not create them, cmd failed, and
' python never started -- reproducing the exact "double-click does nothing"
' symptom this launcher was written to remove. Diagnostics belong in run.py
' (logs/app.log, logs/startup_error.txt), which handles concurrent launches.
'
' Quoting note: nested quotes in VBScript string literals are unreadable and
' easy to get wrong (the first attempt at this file silently launched
' nothing). Q = Chr(34) is used instead so every quote is explicit.

Option Explicit

Dim fso, sh, Q, root, pythonw, runpy, logDir, cmd

Q = Chr(34)
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")

' Resolve everything relative to this script so the folder can be moved.
root = fso.GetParentFolderName(WScript.ScriptFullName)
runpy = fso.BuildPath(root, "run.py")
logDir = fso.BuildPath(root, "logs")
If Not fso.FolderExists(logDir) Then
    fso.CreateFolder(logDir)
End If

If Not fso.FileExists(runpy) Then
    MsgBox "找不到 run.py:" & vbCrLf & runpy, vbCritical, "同声传译 — 启动失败"
    WScript.Quit 1
End If

' Explorer's PATH can differ from a shell's, so prefer the known install and
' only fall back to PATH resolution.
pythonw = sh.ExpandEnvironmentStrings("%LOCALAPPDATA%") & _
          "\Programs\Python\Python312\pythonw.exe"
If Not fso.FileExists(pythonw) Then
    pythonw = "pythonw.exe"
End If

' No cmd, no redirection: both would break a concurrent second launch.
cmd = Q & pythonw & Q & " " & Q & runpy & Q & " --gui"

' Record what we are about to run; if nothing starts, this is the evidence.
On Error Resume Next
Dim f
Set f = fso.CreateTextFile(fso.BuildPath(logDir, "launch_cmd.txt"), True)
f.WriteLine cmd
f.Close
On Error GoTo 0

' 0 = hidden window, False = do not wait
sh.Run cmd, 0, False
