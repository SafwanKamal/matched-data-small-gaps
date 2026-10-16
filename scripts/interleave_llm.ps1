# Let the running decoder seed job finish, run the 4B suite baseline, then resume the remaining seed jobs.
# The trainer writes metrics.json before it prints the "test brier" line, so stopping after that line loses nothing.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$bash = 'C:\Program Files\Git\bin\bash.exe'
$log = Join-Path $root 'logs\interleave.log'
function Note($text) { "$((Get-Date).ToUniversalTime().ToString('HH:mm:ssZ')) $text" | Add-Content -Path $log -Encoding utf8 }

Note 'waiting for seed-20261008/banking77-log_loss to finish'
while (-not (Select-String -Path 'logs\seeds.log' -Pattern 'seed-20261008/banking77-log_loss test brier|SEEDS_EXIT_' -Quiet)) { Start-Sleep -Seconds 30 }

$targets = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_seeds\.sh|after_seeds\.sh|bakeoff\.train_ettin' }
foreach ($target in $targets) { Stop-Process -Id $target.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue }
Note "stopped $(@($targets).Count) seed-chain processes"
Start-Sleep -Seconds 15

$envLine = 'export HF_HOME="$(cygpath -w "$PWD/.cache/hf")" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src PYTHONUTF8=1 PYTHONUNBUFFERED=1; '
Note 'starting Qwen3-4B on the suite'
$llm = Start-Process $bash -PassThru -Wait -WindowStyle Hidden -WorkingDirectory $root -ArgumentList '-c', ('"' + ($envLine + '.venv/Scripts/python.exe -m bakeoff.suite_run llm qwen3-4b-index_prior > logs/qwen4b-suite.log 2>&1').Replace('"', '\"') + '"')
Note "Qwen3-4B run ended with exit code $($llm.ExitCode)"
Start-Sleep -Seconds 15

Note 'resuming the seed chain'
$seeds = Start-Process $bash -PassThru -Wait -WindowStyle Hidden -WorkingDirectory $root -ArgumentList '-c', '"bash scripts/run_seeds.sh > logs/seeds-part2.log 2>&1"'
Note "seed chain ended with exit code $($seeds.ExitCode)"
Note 'INTERLEAVE_DONE'
