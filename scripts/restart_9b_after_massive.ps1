# The running Qwen3.5-9B suite job loaded its settings before CLINC150 was dropped for it.
# Once MASSIVE is written, stop that job and start it again; it resumes from its saved results and skips CLINC150.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root 'logs\restart-9b.log'
function Note($text) { "$((Get-Date).ToUniversalTime().ToString('MM-dd HH:mm:ssZ')) $text" | Add-Content -Path $log -Encoding utf8 }
Note 'waiting for the MASSIVE result'
while (-not (Select-String -Path 'logs\qwen35-9b-suite.log' -Pattern 'qwen3.5-9b-4bit-index_prior massive_en:|SUITE_DONE|Traceback' -Quiet)) { Start-Sleep -Seconds 20 }
$targets = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'suite_run llm qwen3\.5-9b' }
foreach ($target in $targets) { Stop-Process -Id $target.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue }
Note "stopped $(@($targets).Count) processes"
Start-Sleep -Seconds 15
$cmd = 'export HF_HOME="$(cygpath -w "$PWD/.cache/hf")" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1; .venv/Scripts/python.exe -m bakeoff.suite_run llm qwen3.5-9b-4bit-index_prior > logs/qwen35-9b-suite-part2.log 2>&1'
$run = Start-Process 'C:\Program Files\Git\bin\bash.exe' -PassThru -Wait -WindowStyle Hidden -WorkingDirectory $root -ArgumentList '-c', ('"' + $cmd.Replace('"', '\"') + '"')
Note "second part ended with exit code $($run.ExitCode)"
Note 'RESTART_9B_DONE'
