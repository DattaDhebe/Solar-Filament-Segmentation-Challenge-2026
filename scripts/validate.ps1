[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Virtual environment not found. Run .\scripts\setup.ps1 first.'
}

Push-Location $repositoryRoot
try {
    & $python -m ruff format --check .
    if ($LASTEXITCODE -ne 0) {
        throw 'Ruff format check failed.'
    }
    & $python -m ruff check .
    if ($LASTEXITCODE -ne 0) {
        throw 'Ruff lint failed.'
    }
    & $python -m pytest
    if ($LASTEXITCODE -ne 0) {
        throw 'Synthetic tests failed.'
    }

    $annotationPath = Join-Path $repositoryRoot (
        'data\raw\MAGFiLO_1.0_Kaggle_2026\train\' +
        'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    )
    if (Test-Path -LiteralPath $annotationPath) {
        & $python .\scripts\inspect_data.py
        if ($LASTEXITCODE -ne 0) {
            throw 'Downloaded competition data audit failed.'
        }
    }
}
finally {
    Pop-Location
}

Write-Host 'Workspace validation passed.'
