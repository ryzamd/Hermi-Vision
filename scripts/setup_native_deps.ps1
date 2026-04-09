# setup_native_deps.ps1
# Extract TFLite .so from Gradle cache AAR for C++ linking.
# Run ONCE after first Gradle sync.

$ErrorActionPreference = "Stop"

$PROJECT_DIR = Split-Path -Parent $PSScriptRoot
$APP_DIR = Join-Path $PROJECT_DIR "app"
$TFLITE_OUT = Join-Path $APP_DIR "src\main\cpp\third_party\tflite\lib\arm64-v8a"
$GRADLE_CACHE = Join-Path $env:USERPROFILE ".gradle\caches"

Write-Host ""
Write-Host "HermiVision - Setup Native Dependencies" -ForegroundColor Cyan
Write-Host ""

# Step 1: Find TFLite AAR in Gradle cache
Write-Host "[1/3] Searching for TFLite AAR in Gradle cache..." -ForegroundColor Yellow

$aarFiles = Get-ChildItem -Path $GRADLE_CACHE -Recurse -Filter "tensorflow-lite-2.16.1.aar" -ErrorAction SilentlyContinue
if ($aarFiles.Count -eq 0) {
    Write-Host "ERROR: tensorflow-lite-2.16.1.aar not found!" -ForegroundColor Red
    Write-Host "  Run Gradle Sync in Android Studio first." -ForegroundColor Red
    exit 1
}

$aarPath = $aarFiles[0].FullName
Write-Host "  Found: $aarPath" -ForegroundColor Green

# Step 2: Extract libtensorflowlite_jni.so
Write-Host "[2/3] Extracting libtensorflowlite_jni.so..." -ForegroundColor Yellow

New-Item -ItemType Directory -Path $TFLITE_OUT -Force | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($aarPath)
$soEntry = $zip.Entries | Where-Object { $_.FullName -eq "jni/arm64-v8a/libtensorflowlite_jni.so" }

if ($null -eq $soEntry) {
    $zip.Dispose()
    Write-Host "ERROR: .so not found in AAR!" -ForegroundColor Red
    exit 1
}

$destPath = Join-Path $TFLITE_OUT "libtensorflowlite_jni.so"
$stream = $soEntry.Open()
$fileStream = [System.IO.File]::Create($destPath)
$stream.CopyTo($fileStream)
$fileStream.Close()
$stream.Close()
$zip.Dispose()

$fileSizeMB = [math]::Round((Get-Item $destPath).Length / 1048576, 1)
Write-Host "  Extracted: $destPath ($fileSizeMB MB)" -ForegroundColor Green

# Step 3: Verify
Write-Host "[3/3] Verification..." -ForegroundColor Yellow

$headersOk = Test-Path (Join-Path $APP_DIR "src\main\cpp\third_party\tflite\include\tensorflow\lite\c\c_api.h")
$soOk = Test-Path $destPath

if ($headersOk -and $soOk) {
    Write-Host ""
    Write-Host "Setup COMPLETE!" -ForegroundColor Green
    Write-Host "  Headers: OK" -ForegroundColor Green
    Write-Host "  Library: OK ($fileSizeMB MB)" -ForegroundColor Green
    Write-Host "  Now run Gradle Sync again." -ForegroundColor Green
} else {
    Write-Host "WARNING: Missing files!" -ForegroundColor Yellow
    if (-not $headersOk) { Write-Host "  Headers: MISSING" -ForegroundColor Red }
    if (-not $soOk) { Write-Host "  Library: MISSING" -ForegroundColor Red }
}
