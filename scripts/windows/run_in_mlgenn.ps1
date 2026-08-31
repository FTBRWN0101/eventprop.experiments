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
$workingDirectory = Join-Path $repo 'artifacts\genn\live'
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

$environment['Path'] = $visualStudioPath
$environment['DISTUTILS_USE_SDK'] = '1'

# The linker needs LIB, LIBPATH and INCLUDE as well as the compiler on PATH.
# Keep this process-local so launching GeNN cannot mutate the user's shell environment.
foreach ($entry in $environment.GetEnumerator()) {
    [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
}
Set-Location -LiteralPath $workingDirectory
& $python (Resolve-Path -LiteralPath $ScriptPath).Path @ScriptArguments
exit $LASTEXITCODE
