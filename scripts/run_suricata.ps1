#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Runs Suricata against the botnet-detection sample pcap, forcing logs into
  botnet-detection/logs regardless of the current working directory.

  default-log-dir in suricata.yaml is NOT reliably honored by Suricata when
  the path is relative, so we always pass an absolute path via -l.

.NOTES
  Works with PowerShell 7+ (pwsh) on Windows, Linux, and macOS, and with
  Windows PowerShell 5.1.
#>

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir '..')

$Config = Join-Path $RepoRoot 'botnet-detection/config/suricata.yaml'
$Pcap = Join-Path $RepoRoot 'botnet-detection/datasets/opt/Malware-Project/BigDataset/IoTScenarios/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.pcap'
$LogDir = Join-Path $RepoRoot 'botnet-detection/logs'

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

& suricata -c $Config -r $Pcap -l $LogDir @args
exit $LASTEXITCODE
