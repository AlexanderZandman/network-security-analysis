#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Runs Suricata against a pcap file, forcing logs into botnet-detection/logs
  regardless of the current working directory.

  default-log-dir in suricata.yaml is NOT reliably honored by Suricata when
  the path is relative, so we always pass an absolute path via -l.

.PARAMETER Pcap
  Path to the pcap file to analyze. Defaults to the bundled sample pcap.

.PARAMETER ExtraArgs
  Any additional arguments are passed through to suricata.

.NOTES
  Works with PowerShell 7+ (pwsh) on Windows, Linux, and macOS, and with
  Windows PowerShell 5.1.
#>

param(
    [string]$Pcap,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir '..')

$Config = Join-Path $RepoRoot 'botnet-detection/config/suricata.yaml'
$DefaultPcap = Join-Path $RepoRoot 'botnet-detection/datasets/opt/Malware-Project/BigDataset/IoTScenarios/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.pcap'
$LogDir = Join-Path $RepoRoot 'botnet-detection/logs'

if (-not $Pcap) {
    $Pcap = $DefaultPcap
}

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

& suricata -c $Config -r $Pcap -l $LogDir @ExtraArgs
exit $LASTEXITCODE
