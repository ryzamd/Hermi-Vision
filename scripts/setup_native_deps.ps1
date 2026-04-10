# setup_native_deps.ps1
# Extract TFLite .so and delegate headers from Gradle cache AARs for C++ linking.
# Run ONCE after first Gradle sync.

$ErrorActionPreference = "Stop"

$PROJECT_DIR = Split-Path -Parent $PSScriptRoot
$APP_DIR = Join-Path $PROJECT_DIR "app"
$TFLITE_OUT_LIB = Join-Path $APP_DIR "src\main\cpp\third_party\tflite\lib\arm64-v8a"
$TFLITE_OUT_INC = Join-Path $APP_DIR "src\main\cpp\third_party\tflite\include"
$GRADLE_CACHE = Join-Path $env:USERPROFILE ".gradle\caches"

Write-Host ""
Write-Host "HermiVision - Setup Native Dependencies" -ForegroundColor Cyan
Write-Host ""

New-Item -ItemType Directory -Path $TFLITE_OUT_LIB -Force | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem

# ── Helper function ──
function Extract-FromAAR($aarName, $entries) {
    Write-Host "  Searching for $aarName..." -ForegroundColor Yellow
    $aarFiles = Get-ChildItem -Path $GRADLE_CACHE -Recurse -Filter $aarName -ErrorAction SilentlyContinue
    if ($aarFiles.Count -eq 0) {
        Write-Host "  WARNING: $aarName not found in Gradle cache!" -ForegroundColor Red
        return
    }
    $aarPath = $aarFiles[0].FullName
    Write-Host "  Found: $aarPath" -ForegroundColor Green

    $zip = [System.IO.Compression.ZipFile]::OpenRead($aarPath)
    foreach ($spec in $entries) {
        $entry = $zip.Entries | Where-Object { $_.FullName -eq $spec.Source }
        if ($null -ne $entry) {
            $destPath = $spec.Dest
            New-Item -ItemType Directory -Path (Split-Path $destPath) -Force | Out-Null
            $s = $entry.Open()
            $f = [System.IO.File]::Create($destPath)
            $s.CopyTo($f)
            $f.Close()
            $s.Close()
            $sizeMB = [math]::Round((Get-Item $destPath).Length / 1MB, 1)
            Write-Host "    Extracted: $(Split-Path $destPath -Leaf) ($sizeMB MB)" -ForegroundColor Green
        } else {
            Write-Host "    MISSING: $($spec.Source)" -ForegroundColor Red
        }
    }
    $zip.Dispose()
}

# ── Step 1: Base TFLite AAR ──
Write-Host "[1/3] TFLite core..." -ForegroundColor Yellow
Extract-FromAAR "tensorflow-lite-2.16.1.aar" @(
    @{ Source = "jni/arm64-v8a/libtensorflowlite_jni.so";  Dest = "$TFLITE_OUT_LIB\libtensorflowlite_jni.so" },
    @{ Source = "headers/tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"; Dest = "$TFLITE_OUT_INC\tensorflow\lite\delegates\nnapi\nnapi_delegate_c_api.h" }
)

# ── Step 2: GPU Delegate AAR ──
Write-Host "[2/3] GPU delegate..." -ForegroundColor Yellow
Extract-FromAAR "tensorflow-lite-gpu-2.16.1.aar" @(
    @{ Source = "jni/arm64-v8a/libtensorflowlite_gpu_jni.so"; Dest = "$TFLITE_OUT_LIB\libtensorflowlite_gpu_jni.so" },
    @{ Source = "headers/tensorflow/lite/delegates/gpu/delegate.h"; Dest = "$TFLITE_OUT_INC\tensorflow\lite\delegates\gpu\delegate.h" },
    @{ Source = "headers/tensorflow/lite/delegates/gpu/delegate_options.h"; Dest = "$TFLITE_OUT_INC\tensorflow\lite\delegates\gpu\delegate_options.h" }
)

# ── Step 3: Verify ──
Write-Host "[3/3] Verification..." -ForegroundColor Yellow

$checks = @(
    @{ Name = "TFLite .so";      Path = "$TFLITE_OUT_LIB\libtensorflowlite_jni.so" },
    @{ Name = "GPU .so";         Path = "$TFLITE_OUT_LIB\libtensorflowlite_gpu_jni.so" },
    @{ Name = "C API header";    Path = "$TFLITE_OUT_INC\tensorflow\lite\c\c_api.h" },
    @{ Name = "GPU header";      Path = "$TFLITE_OUT_INC\tensorflow\lite\delegates\gpu\delegate.h" },
    @{ Name = "NNAPI header";    Path = "$TFLITE_OUT_INC\tensorflow\lite\delegates\nnapi\nnapi_delegate_c_api.h" },
    @{ Name = "XNNPACK header";  Path = "$TFLITE_OUT_INC\tensorflow\lite\delegates\xnnpack\xnnpack_delegate.h" }
)

$allOk = $true
foreach ($c in $checks) {
    if (Test-Path $c.Path) {
        Write-Host "  $($c.Name): OK" -ForegroundColor Green
    } else {
        Write-Host "  $($c.Name): MISSING" -ForegroundColor Red
        $allOk = $false
    }
}

# ── Step 4: Clean stale build caches (prevent R.jar lock on next build) ──
Write-Host "[4/4] Preparing for Android Studio build..." -ForegroundColor Yellow

# Stop CLI Gradle daemons
$gradlew = Join-Path $PROJECT_DIR "gradlew.bat"
if (Test-Path $gradlew) {
    Write-Host "  Stopping Gradle daemons (CLI)..." -ForegroundColor Gray
    & $gradlew --stop 2>$null | Out-Null
}

# Kill ALL Gradle daemon java processes (Android Studio daemons)
Write-Host "  Stopping Gradle daemons (Background)..." -ForegroundColor Gray

# Method 1: Get-CimInstance (Modern Windows)
$daemons = Get-CimInstance Win32_Process -Filter "Name='java.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "GradleDaemon" }
if ($daemons) {
    # Ensure it's an array to iterate safely
    @($daemons) | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

# Method 2: WMIC (Legacy Windows fallback)
if (Get-Command "wmic" -ErrorAction SilentlyContinue) {
    try {
        & wmic process where "name='java.exe' and commandline like '%GradleDaemon%'" call terminate 2>$null | Out-Null
    } catch {}
}
Start-Sleep -Seconds 1 # Wait for Windows to release file handles

# Clean CMake cache (forces reconfigure with current flags)
$cxxDir = Join-Path $APP_DIR ".cxx"
if (Test-Path $cxxDir) {
    Remove-Item -Recurse -Force $cxxDir
    Write-Host "  Cleaned .cxx CMake cache" -ForegroundColor Green
}

# Clean locked R.jar intermediates (retry if still locked)
$rJarDir = Join-Path $APP_DIR "build\intermediates\compile_and_runtime_not_namespaced_r_class_jar"
if (Test-Path $rJarDir) {
    Start-Sleep -Milliseconds 500
    Remove-Item -Recurse -Force $rJarDir -ErrorAction SilentlyContinue
    Write-Host "  Cleaned R.jar intermediates" -ForegroundColor Green
}

Write-Host ""
if ($allOk) {
    Write-Host "Setup COMPLETE! Ready to build in Android Studio." -ForegroundColor Green
} else {
    Write-Host "Setup INCOMPLETE. Some files are missing." -ForegroundColor Yellow
}
