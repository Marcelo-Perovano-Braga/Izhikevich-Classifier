[CmdletBinding()]
param(
    [switch]$SkipDependencyInstall,
    [switch]$SkipArchive
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectRoot

try {
    $PythonCommand = Get-Command python -ErrorAction Stop
    $PythonExe = $PythonCommand.Source

    $Platform = (& $PythonExe -c "import sys; print(sys.platform)").Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Could not query the Python interpreter."
    }
    if ($Platform -ne "win32") {
        throw "The .exe must be built on Windows."
    }

    $PointerBits = (& $PythonExe -c "import struct; print(struct.calcsize('P') * 8)").Trim()
    if ($LASTEXITCODE -ne 0 -or $PointerBits -ne "64") {
        throw "Use a 64-bit Python installation to build the application."
    }

    if (-not (Test-Path "weights\classifier.pt" -PathType Leaf)) {
        throw (
            "Missing checkpoint at weights\classifier.pt. " +
            "Run train_pipeline.py or copy the trained checkpoint."
        )
    }

    if (-not $SkipDependencyInstall) {
        Write-Host "Installing application dependencies..." -ForegroundColor Cyan
        & $PythonExe -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install requirements.txt."
        }

        Write-Host "Installing packaging dependencies..." -ForegroundColor Cyan
        & $PythonExe -m pip install -r requirements-build.txt
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install requirements-build.txt."
        }
    }

    Write-Host "Building the Windows application..." -ForegroundColor Cyan
    & $PythonExe -m PyInstaller --clean --noconfirm IzhikevichClassifier.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller did not finish building the application."
    }

    $DistributionDirectory = Join-Path $ProjectRoot "dist\IzhikevichClassifier"
    $ExecutablePath = Join-Path $DistributionDirectory "IzhikevichClassifier.exe"
    if (-not (Test-Path $ExecutablePath -PathType Leaf)) {
        throw "Build finished without producing $ExecutablePath."
    }

    if (-not $SkipArchive) {
        $ArchivePath = Join-Path $ProjectRoot "dist\IzhikevichClassifier-Windows-x64.zip"
        if (Test-Path $ArchivePath -PathType Leaf) {
            Remove-Item $ArchivePath -Force
        }
        Write-Host "Creating the distribution archive..." -ForegroundColor Cyan
        Compress-Archive -Path "$DistributionDirectory\*" -DestinationPath $ArchivePath
        Write-Host "Package: $ArchivePath" -ForegroundColor Green
    }

    $Hash = (Get-FileHash $ExecutablePath -Algorithm SHA256).Hash
    Write-Host "Executable: $ExecutablePath" -ForegroundColor Green
    Write-Host "SHA-256: $Hash" -ForegroundColor Green
    Write-Host (
        "Keep the executable inside the IzhikevichClassifier folder; " +
        "dependencies are in the _internal subdirectory."
    ) -ForegroundColor Yellow
}
finally {
    Pop-Location
}
