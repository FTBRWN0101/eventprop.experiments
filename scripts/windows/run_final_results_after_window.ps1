param(
    [Parameter(Mandatory = $true)]
    [int]$WindowProcessId
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$manifestPath = Join-Path $taskRoot 'artifacts\results\runs\final_weekly_20260831_v3_window_f010_L45_manifest.json'
$relocatedManifestPath = Join-Path $taskRoot 'artifacts\genn\live\artifacts\results\runs\final_weekly_20260831_v3_window_f010_L45_manifest.json'

try {
    Wait-Process -Id $WindowProcessId -ErrorAction Stop
} catch [System.ArgumentException] {
    #The window process may have finished between dispatch and this script starting.
}

if (-not (Test-Path $manifestPath) -and (Test-Path $relocatedManifestPath)) {
    Copy-Item -LiteralPath $relocatedManifestPath -Destination $manifestPath
}
if (-not (Test-Path $manifestPath)) {
    throw "Window replication did not create its manifest: $manifestPath"
}
$records = @(Get-Content $manifestPath -Raw | ConvertFrom-Json)
$failed = @($records | Where-Object { $_.returncode -ne 0 })
if ($records.Count -ne 40 -or $failed.Count -ne 0) {
    throw "Window replication is incomplete: $($records.Count)/40 records, $($failed.Count) failed. Selected-results work was not started."
}

& cmd.exe /d /c 'scripts\windows\run_final_selected_results.bat'
exit $LASTEXITCODE
