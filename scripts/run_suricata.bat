@echo off
REM Runs Suricata against a pcap file, forcing logs into
REM botnet-detection\logs regardless of the current working directory.
REM
REM default-log-dir in suricata.yaml is NOT reliably honored by Suricata when
REM the path is relative, so we always pass an absolute path via -l.
REM
REM Usage: run_suricata.bat [PCAP_FILE] [extra suricata args...]
REM   PCAP_FILE defaults to the bundled sample pcap if omitted.

setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "REPO_ROOT=%%~fI"

set "CONFIG=%REPO_ROOT%\botnet-detection\config\suricata.yaml"
set "DEFAULT_PCAP=%REPO_ROOT%\botnet-detection\datasets\opt\Malware-Project\BigDataset\IoTScenarios\CTU-IoT-Malware-Capture-1-1\2018-05-09-192.168.100.103.pcap"
set "LOG_DIR=%REPO_ROOT%\botnet-detection\logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if "%~1"=="" (
    set "PCAP=%DEFAULT_PCAP%"
) else (
    set "PCAP=%~1"
    shift
)

REM NOTE: %* does not reflect `shift` in cmd.exe, so remaining args are
REM rebuilt manually from %1 onward (which *is* updated by shift).
set "EXTRA_ARGS="
:collect_args
if not "%~1"=="" (
    set "EXTRA_ARGS=%EXTRA_ARGS% %1"
    shift
    goto collect_args
)

suricata -c "%CONFIG%" -r "%PCAP%" -l "%LOG_DIR%" %EXTRA_ARGS%

endlocal
