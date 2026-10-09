@echo off
REM Runs Suricata against the botnet-detection sample pcap, forcing logs into
REM botnet-detection\logs regardless of the current working directory.
REM
REM default-log-dir in suricata.yaml is NOT reliably honored by Suricata when
REM the path is relative, so we always pass an absolute path via -l.

setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "REPO_ROOT=%%~fI"

set "CONFIG=%REPO_ROOT%\botnet-detection\config\suricata.yaml"
set "PCAP=%REPO_ROOT%\botnet-detection\datasets\opt\Malware-Project\BigDataset\IoTScenarios\CTU-IoT-Malware-Capture-1-1\2018-05-09-192.168.100.103.pcap"
set "LOG_DIR=%REPO_ROOT%\botnet-detection\logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

suricata -c "%CONFIG%" -r "%PCAP%" -l "%LOG_DIR%" %*

endlocal
