Set-StrictMode -Version Latest

function Resolve-KaggleExecutable {
    param(
        [string]$KaggleExecutable
    )

    if ($KaggleExecutable) {
        $resolved = Resolve-Path -LiteralPath $KaggleExecutable -ErrorAction Stop
        return $resolved.Path
    }

    $localExecutable = Join-Path $PSScriptRoot '..\.venv\Scripts\kaggle.exe'
    if (Test-Path -LiteralPath $localExecutable) {
        return (Resolve-Path -LiteralPath $localExecutable).Path
    }

    $command = Get-Command kaggle -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw 'Kaggle CLI not found. Run .\scripts\setup.ps1 first.'
}
