[CmdletBinding()]
param(
    [string]$PythonExecutable = 'python'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvPython = Join-Path $repositoryRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $PythonExecutable -m venv (Join-Path $repositoryRoot '.venv')
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the Python virtual environment.'
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to upgrade pip.'
}

& $venvPython -m pip install -e "$repositoryRoot[dev]"
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install project dependencies.'
}

Write-Host "Environment ready: $venvPython"
