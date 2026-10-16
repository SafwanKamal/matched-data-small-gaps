# Wait for one seed job to finish, stop the seed chain, run one GPU command, then resume the seeds.
# The trainer writes metrics.json before it prints the "test brier" line, so stopping after that line loses nothing.
param(
    [Parameter(Mandatory = $true)][string]$WaitLog,
    [Parameter(Mandatory = $true)][string]$WaitPattern,
    [Parameter(Mandatory = $true)][string]$Module,
    [Parameter(Mandatory = $true)][string]$ModuleLog,
    [Parameter(Mandatory = $true)][string]$ResumeLog
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$bash = 'C:\Program Files\Git\bin\bash.exe'
$log = Join-Path $root 'logs\slot-in.log'
function Note($text) { "$((Get-Date).ToUniversalTime().ToString('MM-dd HH:mm:ssZ')) $text" | Add-Content -Path $log -Encoding utf8 }

Note "waiting for '$WaitPattern' in $WaitLog"
while (-not (Select-String -Path $WaitLog -Pattern "$WaitPattern|SEEDS_EXIT_" -Quiet)) { Start-Sleep -Seconds 30 }

$targets = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_seeds\.sh|after_seeds\.sh|interleave_llm\.ps1|bakeoff\.train_ettin' }
foreach ($target in $targets) { Stop-Process -Id $target.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue }
Note "stopped $(@($targets).Count) seed-chain processes"
Start-Sleep -Seconds 15

$envLine = 'export HF_HOME="$(cygpath -w "$PWD/.cache/hf")" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1; '
Note "starting python -m $Module"
$run = Start-Process $bash -PassThru -Wait -WindowStyle Hidden -WorkingDirectory $root -ArgumentList '-c', ('"' + ($envLine + ".venv/Scripts/python.exe -m $Module > $ModuleLog 2>&1").Replace('"', '\"') + '"')
Note "python -m $Module ended with exit code $($run.ExitCode)"
Start-Sleep -Seconds 15

Note 'resuming the seed chain'
$seeds = Start-Process $bash -PassThru -Wait -WindowStyle Hidden -WorkingDirectory $root -ArgumentList '-c', ('"bash scripts/run_seeds.sh > ' + $ResumeLog + ' 2>&1"')
Note "seed chain ended with exit code $($seeds.ExitCode)"
Note 'SLOT_IN_DONE'
