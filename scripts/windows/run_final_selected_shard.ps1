[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$ShardIndex,

    [ValidateRange(2, 8)]
    [int]$ShardCount = 2,

    [switch]$FinalizeOnly,

    [switch]$DryRun,

    [string]$OutputRoot = 'artifacts\results\selected-parallel-v1'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$resultsRoot = [System.IO.Path]::GetFullPath((Join-Path $repo 'artifacts\results'))
$requestedOutput = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot
}
else {
    Join-Path $repo $OutputRoot
}
$outRoot = [System.IO.Path]::GetFullPath($requestedOutput)
$resultsPrefix = $resultsRoot.TrimEnd('\') + '\'
if (-not $outRoot.StartsWith($resultsPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must remain under artifacts\\results: $outRoot"
}
if ($ShardIndex -lt 0 -or $ShardIndex -ge $ShardCount) {
    throw "ShardIndex must be in 0..$($ShardCount - 1)"
}

$materializer = Join-Path $repo 'src\experiments\tools\materialize_selection.py'
$runner = Join-Path $PSScriptRoot 'run_in_mlgenn.ps1'
$arguments = @(
    '--campaign-id', 'final_weekly_20260831_v3_holdout',
    '--campaign-id', 'final_weekly_20260831_v3_window_f010_L45',
    '--selection-metric', 'val_r2',
    '--rate-draws', '5',
    '--out-root', $outRoot,
    '--shard-index', $ShardIndex,
    '--shard-count', $ShardCount
)
if ($FinalizeOnly) {
    $arguments += '--finalize-only'
}
else {
    #Each worker owns its CWD because PyGeNN writes generated C++ under it.
    $env:GENN_BUILD = Join-Path $repo "artifacts\genn\selected-workers\worker-$ShardIndex"
}
if ($DryRun) {
    $arguments += '--dry-run'
}

& $runner -ScriptPath $materializer -ScriptArguments $arguments
exit $LASTEXITCODE
