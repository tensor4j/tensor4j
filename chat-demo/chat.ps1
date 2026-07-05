# Interactive chat via tensor4j.
# Default: Qwen2.5-1.5B Instruct Q4_K_M (~1 GB, bigger than Llama 3.2 1B).
# Use -Llama / --llama for Llama 3.2 1B Instruct instead.
#
# Usage:
#   .\chat.ps1
#   .\chat.ps1 -Download              # fetch default Qwen GGUF if missing
#   .\chat.ps1 -Llama                 # Llama 3.2 1B instead of Qwen
#   .\chat.ps1 -Gguf C:\path\to\model.gguf
#   $env:TENSOR4J_CHAT_MAX_TOKENS = "256"; .\chat.ps1
#
# Options (also available as env vars):
#   -Llama / --llama     use Llama 3.2 1B (template llama3, log base llama32)
#   -Download / --download   download default Qwen GGUF from Hugging Face
#   -Gguf / --gguf PATH   TENSOR4J_GGUF_PATH
#   -Template / --template NAME   TENSOR4J_CHAT_TEMPLATE
#   -Mode / --mode MODE   TENSOR4J_CHAT_MODE (quality|greedy)
#   -MaxTokens / --max-tokens N   TENSOR4J_CHAT_MAX_TOKENS
#   -SaveDir / --save-dir PATH   TENSOR4J_CHAT_SAVE_DIR
#   -Build / --build         run mvn install before chat (default: skip build)
#   -SkipBuild / --skip-build   default; kept for compatibility
#   -Help / --help / -h          show help

$ErrorActionPreference = "Stop"

function Show-Usage {
    Get-Content $PSCommandPath | Select-Object -Skip 1 -First 22 | ForEach-Object {
        if ($_ -match '^\# ?(.*)$') { $Matches[1] }
    }
    exit 0
}

$UseLlama = $false
$Download = $false
$SkipBuild = $true
$ShowHelp = $false

$i = 0
while ($i -lt $args.Count) {
    switch ($args[$i]) {
        { $_ -in '-Llama', '--llama' } { $UseLlama = $true; $i++; continue }
        { $_ -in '-Download', '--download' } { $Download = $true; $i++; continue }
        { $_ -in '-Build', '--build' } { $SkipBuild = $false; $i++; continue }
        { $_ -in '-SkipBuild', '--skip-build' } { $SkipBuild = $true; $i++; continue }
        { $_ -in '-Help', '--help', '-h' } { $ShowHelp = $true; $i++; continue }
        { $_ -in '-Gguf', '--gguf' } {
            if ($i + 1 -ge $args.Count) { Write-Error "$($args[$i]) requires a path" }
            $env:TENSOR4J_GGUF_PATH = $args[$i + 1]
            $i += 2
            continue
        }
        { $_ -in '-Template', '--template' } {
            if ($i + 1 -ge $args.Count) { Write-Error "$($args[$i]) requires a value" }
            $env:TENSOR4J_CHAT_TEMPLATE = $args[$i + 1]
            $i += 2
            continue
        }
        { $_ -in '-Mode', '--mode' } {
            if ($i + 1 -ge $args.Count) { Write-Error "$($args[$i]) requires a value" }
            $env:TENSOR4J_CHAT_MODE = $args[$i + 1]
            $i += 2
            continue
        }
        { $_ -in '-MaxTokens', '--max-tokens' } {
            if ($i + 1 -ge $args.Count) { Write-Error "$($args[$i]) requires a value" }
            $env:TENSOR4J_CHAT_MAX_TOKENS = $args[$i + 1]
            $i += 2
            continue
        }
        { $_ -in '-SaveDir', '--save-dir' } {
            if ($i + 1 -ge $args.Count) { Write-Error "$($args[$i]) requires a value" }
            $env:TENSOR4J_CHAT_SAVE_DIR = $args[$i + 1]
            $i += 2
            continue
        }
        { $_ -in '-Legacy', '--legacy' } {
            $env:TENSOR4J_CHAT_HISTORY_MODE = "legacy"
            $i += 1
            continue
        }
        default {
            Write-Error "Unknown option: $($args[$i])"
        }
    }
}

if ($ShowHelp) { Show-Usage }

$ScriptDir = $PSScriptRoot
$Tensor4jRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$HomeDir = if ($env:USERPROFILE) { $env:USERPROFILE } else { $env:HOME }
$ModelsDir = Join-Path $HomeDir ".local/models"

if (-not $env:TENSOR4J_CHAT_MODE) { $env:TENSOR4J_CHAT_MODE = "quality" }
if (-not $env:TENSOR4J_CHAT_HISTORY_MODE) { $env:TENSOR4J_CHAT_HISTORY_MODE = "llama" }
if (-not $env:TENSOR4J_CHAT_MAX_TOKENS) { $env:TENSOR4J_CHAT_MAX_TOKENS = "256" }
if (-not $env:TENSOR4J_CHAT_SAVE_DIR) {
    $env:TENSOR4J_CHAT_SAVE_DIR = Join-Path $HomeDir ".local/conversations"
}

function Find-FirstExistingFile {
    param([string[]]$Candidates)
    foreach ($path in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($path)) { continue }
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    return $null
}

function Expand-SnapshotGguf {
    param([string]$SnapshotsDir, [string]$FileName)
    if (-not (Test-Path -LiteralPath $SnapshotsDir)) { return @() }
    $found = @()
    Get-ChildItem -LiteralPath $SnapshotsDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $candidate = Join-Path $_.FullName $FileName
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $found += $candidate
        }
    }
    return $found
}

function Find-LlamaGguf {
    $candidates = @()
    if ($env:TENSOR4J_GGUF_PATH) { $candidates += $env:TENSOR4J_GGUF_PATH }
    $hfRoot = Join-Path $HomeDir ".cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots"
    $candidates += Expand-SnapshotGguf $hfRoot "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    $candidates += Join-Path $HomeDir "AppData/Local/Temp/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    return Find-FirstExistingFile $candidates
}

function Find-QwenGguf {
    $qwenFile = if ($env:TENSOR4J_QWEN_GGUF_FILE) { $env:TENSOR4J_QWEN_GGUF_FILE } else { "qwen2.5-1.5b-instruct-q4_k_m.gguf" }
    $candidates = @()
    if ($env:TENSOR4J_GGUF_PATH) { $candidates += $env:TENSOR4J_GGUF_PATH }
    if ($env:TENSOR4J_QWEN_GGUF) { $candidates += $env:TENSOR4J_QWEN_GGUF }
    $candidates += Join-Path $ModelsDir $qwenFile
    $candidates += Join-Path $ModelsDir "qwen2.5-3b-instruct-q4_k_m.gguf"
    $hf15 = Join-Path $HomeDir ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct-GGUF/snapshots"
    $candidates += Expand-SnapshotGguf $hf15 $qwenFile
    $hf3 = Join-Path $HomeDir ".cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct-GGUF/snapshots"
    $candidates += Expand-SnapshotGguf $hf3 "qwen2.5-3b-instruct-q4_k_m.gguf"
    return Find-FirstExistingFile $candidates
}

function Get-HfCli {
    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if ($hf) { return $hf.Source }
    $cli = Get-Command huggingface-cli -ErrorAction SilentlyContinue
    if ($cli) { return $cli.Source }
    return $null
}

function Download-QwenGguf {
    $hfCli = Get-HfCli
    if (-not $hfCli) {
        Write-Error "Qwen GGUF not found. Install huggingface_hub and run: .\chat.ps1 -Download`n  pip install huggingface_hub"
    }
    $repo = if ($env:TENSOR4J_QWEN_REPO) { $env:TENSOR4J_QWEN_REPO } else { "Qwen/Qwen2.5-1.5B-Instruct-GGUF" }
    $file = if ($env:TENSOR4J_QWEN_GGUF_FILE) { $env:TENSOR4J_QWEN_GGUF_FILE } else { "qwen2.5-1.5b-instruct-q4_k_m.gguf" }
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
    Write-Host "Downloading ${repo} ${file} (~1 GB)..." -ForegroundColor Yellow
    if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = "utf-8" }
    & $hfCli download $repo $file --local-dir $ModelsDir
    if ($LASTEXITCODE -ne 0) { throw "hf download failed with exit code $LASTEXITCODE" }
    $env:TENSOR4J_GGUF_PATH = Join-Path $ModelsDir $file
}

if ($UseLlama) {
    if (-not $env:TENSOR4J_CHAT_TEMPLATE) { $env:TENSOR4J_CHAT_TEMPLATE = "llama3" }
    if (-not $env:TENSOR4J_CHAT_LOG_BASE) { $env:TENSOR4J_CHAT_LOG_BASE = "llama32" }
    if (-not $env:TENSOR4J_GGUF_PATH) {
        $found = Find-LlamaGguf
        if ($found) { $env:TENSOR4J_GGUF_PATH = $found }
    }
    if (-not $env:TENSOR4J_GGUF_PATH -or -not (Test-Path -LiteralPath $env:TENSOR4J_GGUF_PATH)) {
        Write-Error "Llama GGUF not found. Set TENSOR4J_GGUF_PATH or pass -Gguf PATH`n  example: .\chat.ps1 -Llama -Gguf ~\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    }
    Write-Host "Using Llama GGUF: $($env:TENSOR4J_GGUF_PATH)" -ForegroundColor Cyan
} else {
    if (-not $env:TENSOR4J_CHAT_TEMPLATE) { $env:TENSOR4J_CHAT_TEMPLATE = "qwen2" }
    if (-not $env:TENSOR4J_CHAT_LOG_BASE) { $env:TENSOR4J_CHAT_LOG_BASE = "qwen25-1.5b" }
    if ($Download) {
        Download-QwenGguf
    } elseif (-not $env:TENSOR4J_GGUF_PATH) {
        $found = Find-QwenGguf
        if ($found) { $env:TENSOR4J_GGUF_PATH = $found }
    }
    if (-not $env:TENSOR4J_GGUF_PATH -or -not (Test-Path -LiteralPath $env:TENSOR4J_GGUF_PATH)) {
        Write-Error "Qwen GGUF not found. Run: .\chat.ps1 -Download"
    }
    Write-Host "Using Qwen GGUF: $($env:TENSOR4J_GGUF_PATH)" -ForegroundColor Cyan
}

New-Item -ItemType Directory -Force -Path $env:TENSOR4J_CHAT_SAVE_DIR | Out-Null

if (-not $SkipBuild) {
    Write-Host "Building tensor4j + chat-demo..." -ForegroundColor Yellow
    Push-Location $Tensor4jRoot
    try {
        mvn -q install -DskipTests
        if ($LASTEXITCODE -ne 0) { throw "mvn install failed" }
    } finally {
        Pop-Location
    }
    Push-Location $ScriptDir
    try {
        mvn -q package -DskipTests
        if ($LASTEXITCODE -ne 0) { throw "mvn package failed" }
    } finally {
        Pop-Location
    }
}

if ($env:MAVEN_OPTS) {
    if ($env:MAVEN_OPTS -notmatch '-Xmx') {
        $env:MAVEN_OPTS = "$($env:MAVEN_OPTS) -Xmx10g"
    }
} else {
    $env:MAVEN_OPTS = "-Xmx10g"
}

Push-Location $ScriptDir
try {
    mvn -q exec:java -f (Join-Path $ScriptDir "pom.xml")
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
