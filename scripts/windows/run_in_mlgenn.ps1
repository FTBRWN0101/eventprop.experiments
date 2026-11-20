[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$ScriptPath,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArguments
)

$ErrorActionPreference = 'Stop'
trap {
    Write-Error "run_in_mlgenn.ps1 failed on line $($_.InvocationInfo.ScriptLineNumber): $($_.Exception.Message)"
    exit 1
}

$scriptDirectory = Split-Path -Parent $PSCommandPath
$repo = (Resolve-Path (Join-Path $scriptDirectory '..\..')).Path
$defaultBuildDirectory = Join-Path $repo 'artifacts\genn\live'
$workingDirectory = if ([string]::IsNullOrWhiteSpace($env:GENN_BUILD)) {
    $defaultBuildDirectory
}
else {
    [System.IO.Path]::GetFullPath($env:GENN_BUILD)
}
$python = 'C:\Users\user\miniconda3\envs\mlgenn\python.exe'
$vcvarsAll = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "mlgenn Python was not found at $python"
}
if (-not (Test-Path -LiteralPath $vcvarsAll -PathType Leaf)) {
    throw "Visual Studio environment setup was not found at $vcvarsAll"
}
if (-not (Test-Path -LiteralPath $workingDirectory -PathType Container)) {
    New-Item -ItemType Directory -Force $workingDirectory | Out-Null
}
if (-not $workingDirectory.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "GENN_BUILD must remain inside the repository: $workingDirectory"
}

# vcvarsall exposes both PATH and Path here. Keep the candidate that contains the
# Visual Studio build tools, otherwise setuptools can erase MSBuild before PyGeNN imports.
$environmentLines = & cmd.exe /d /c ('call "{0}" x64 >nul && set' -f $vcvarsAll)
if ($LASTEXITCODE -ne 0) {
    throw 'vcvarsall failed to configure the x64 build environment'
}

$environment = [ordered]@{}
$fallbackPath = $null
$visualStudioPath = $null
foreach ($line in $environmentLines) {
    if ($line -notmatch '^([^=]+)=(.*)$') {
        continue
    }

    $name = $Matches[1]
    $value = $Matches[2]
    if ($name -ieq 'Path') {
        if ($value -match '[\\/]MSBuild[\\/]Current[\\/]Bin') {
            $visualStudioPath = $value
        }
        elseif ($null -eq $fallbackPath) {
            $fallbackPath = $value
        }
        continue
    }
    $environment[$name] = $value
}

if ($null -eq $visualStudioPath) {
    $visualStudioPath = $fallbackPath
}
if ([string]::IsNullOrWhiteSpace($visualStudioPath)) {
    throw 'vcvarsall did not provide a PATH value'
}

$compilerDirectory = @($visualStudioPath -split ';' | Where-Object {
    -not [string]::IsNullOrWhiteSpace($_) -and
    (Test-Path -LiteralPath (Join-Path $_ 'cl.exe') -PathType Leaf)
}) | Select-Object -First 1
if ($null -eq $compilerDirectory) {
    throw 'vcvarsall did not provide a usable cl.exe host compiler'
}

$environment['Path'] = $visualStudioPath
# CUDA 13.3 does not reliably discover cl.exe from a PowerShell-launched child,
# even when vcvarsall has populated PATH. Point nvcc at the exact compiler instead.
$environment['CUDAHOSTCXX'] = Join-Path $compilerDirectory 'cl.exe'
$environment['DISTUTILS_USE_SDK'] = '1'

# The linker needs LIB, LIBPATH and INCLUDE as well as the compiler on PATH.
# Keep this process-local so launching GeNN cannot mutate the user's shell environment.
foreach ($entry in $environment.GetEnumerator()) {
    [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
}
Set-Location -LiteralPath $workingDirectory
$resolvedScript = (Resolve-Path -LiteralPath $ScriptPath).Path
$quoteForCmd = {
    param([string]$Value)
    '"' + $Value.Replace('"', '""') + '"'
}
$command = @(
    (& $quoteForCmd $python)
    (& $quoteForCmd $resolvedScript)
    @($ScriptArguments | ForEach-Object { & $quoteForCmd $_ })
) -join ' '

# cmd.exe receives the process environment that vcvarsall configured above. A direct
# PowerShell executable launch on this host restores Python's original PATH, which
# leaves nvcc unable to locate its MSVC host compiler.
& cmd.exe /d /s /c $command
exit $LASTEXITCODE
