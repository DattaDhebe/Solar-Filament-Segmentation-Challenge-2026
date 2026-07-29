[CmdletBinding()]
param(
    [string]$Competition = 'filament-segmentation-2026',
    [string]$Destination = 'data\raw',
    [string]$KaggleExecutable = '',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'common.ps1')

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$destinationPath = if ([System.IO.Path]::IsPathRooted($Destination)) {
    [System.IO.Path]::GetFullPath($Destination)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $Destination))
}
$expectedRoot = Join-Path $destinationPath 'MAGFiLO_1.0_Kaggle_2026'

if ((Test-Path -LiteralPath $expectedRoot) -and -not $Force) {
    Write-Host "Dataset already exists: $expectedRoot"
}
else {
    New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
    $kaggle = Resolve-KaggleExecutable -KaggleExecutable $KaggleExecutable
    $downloadArguments = @(
        'competitions',
        'download',
        '-c',
        $Competition,
        '-p',
        $destinationPath
    )
    if ($Force) {
        $downloadArguments += '--force'
    }

    & $kaggle @downloadArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Competition data download failed.'
    }

    $archive = Get-ChildItem -LiteralPath $destinationPath -Filter '*.zip' |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $archive) {
        throw "No downloaded ZIP archive was found in $destinationPath."
    }

    $expandArguments = @{
        LiteralPath = $archive.FullName
        DestinationPath = $destinationPath
    }
    if ($Force) {
        $expandArguments['Force'] = $true
    }
    Expand-Archive @expandArguments
}

$trainDirectory = Join-Path $expectedRoot 'train\train_images'
$testDirectory = Join-Path $expectedRoot 'test\test_images'
$annotationPath = Join-Path $expectedRoot 'train\MAGFiLO_1.0_Annotations_kaggle2026_train.json'
if (-not (Test-Path -LiteralPath $annotationPath)) {
    throw "Missing annotation file: $annotationPath"
}

$trainCount = @(Get-ChildItem -LiteralPath $trainDirectory -File -Filter '*.jpeg').Count
$testCount = @(Get-ChildItem -LiteralPath $testDirectory -File -Filter '*.jpeg').Count
if ($trainCount -ne 707 -or $testCount -ne 180) {
    throw "Unexpected image inventory: train=$trainCount, test=$testCount."
}

Write-Host "Dataset ready: $expectedRoot"
Write-Host "Verified images: train=$trainCount, test=$testCount"
