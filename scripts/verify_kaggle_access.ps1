[CmdletBinding()]
param(
    [string]$Competition = 'filament-segmentation-2026',
    [string]$KaggleExecutable = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'common.ps1')

$kaggle = Resolve-KaggleExecutable -KaggleExecutable $KaggleExecutable

Write-Host "Checking competition entry and submission allowance: $Competition"
& $kaggle competitions submission-limits $Competition --json
if ($LASTEXITCODE -ne 0) {
    throw 'Kaggle did not confirm competition access. Confirm that the rules were accepted.'
}

Write-Host 'Checking the official data endpoint without downloading data.'
$null = & $kaggle competitions files -c $Competition --page-size 1 --quiet 2>&1
if ($LASTEXITCODE -ne 0) {
    throw 'Kaggle data listing failed.'
}

Write-Host 'Kaggle access verified. No credential file was opened or changed by this script.'
