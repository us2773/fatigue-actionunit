param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,

    [Parameter(Mandatory = $true)]
    [string]$OutputCsv,

    [Parameter(Mandatory = $true)]
    [string]$LocalEnvironmentConfig,

    [Parameter(Mandatory = $true)]
    [string]$RunId,

    [switch]$Force,

    [string]$FeatureExtractionPath = "/home/openface-build/build/bin/FeatureExtraction",

    [string]$ContainerWorkRoot = "/tmp/fatigue-actionunit",

    [switch]$KeepContainerWorkDir
)

$ErrorActionPreference = "Stop"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "External command failed: $FilePath"
    }
}

function Quote-Sh {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + ($Value -replace "'", "'`"`"'") + "'"
}

if (-not (Test-Path -LiteralPath $InputVideo)) {
    throw "Input video not found."
}

if ((Test-Path -LiteralPath $OutputCsv) -and (-not $Force)) {
    Write-Output "skip_existing"
    exit 0
}

if (-not (Test-Path -LiteralPath $LocalEnvironmentConfig)) {
    throw "Local OpenFace environment config not found."
}

. $LocalEnvironmentConfig

$ContainerRef = $null
if ($ContainerName) {
    $ContainerRef = $ContainerName
}
elseif ($ContainerID) {
    $ContainerRef = $ContainerID
}
else {
    throw "ContainerName or ContainerID is required in local environment config."
}

$InputVideoPath = Resolve-Path -LiteralPath $InputVideo
$OutputCsvPath = [System.IO.Path]::GetFullPath($OutputCsv)
$OutputDirectory = [System.IO.Path]::GetDirectoryName($OutputCsvPath)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$Stem = [System.IO.Path]::GetFileNameWithoutExtension($InputVideoPath)
$ContainerRunDir = "$ContainerWorkRoot/$RunId/$Stem"
$ContainerInputDir = "$ContainerRunDir/input"
$ContainerOutputDir = "$ContainerRunDir/output"
$ContainerVideoPath = "$ContainerInputDir/$Stem.mp4"
$ContainerCsvPath = "$ContainerOutputDir/$Stem.csv"

$SetupCommand = "mkdir -p $(Quote-Sh $ContainerInputDir) $(Quote-Sh $ContainerOutputDir)"
Invoke-Native -FilePath "docker" -Arguments @("exec", $ContainerRef, "sh", "-lc", $SetupCommand)
Invoke-Native -FilePath "docker" -Arguments @("cp", $InputVideoPath.Path, "${ContainerRef}:$ContainerVideoPath")

$FeatureCommand = "$(Quote-Sh $FeatureExtractionPath) -f $(Quote-Sh $ContainerVideoPath) -out_dir $(Quote-Sh $ContainerOutputDir) -aus"
Invoke-Native -FilePath "docker" -Arguments @("exec", $ContainerRef, "sh", "-lc", $FeatureCommand)
Invoke-Native -FilePath "docker" -Arguments @("cp", "${ContainerRef}:$ContainerCsvPath", $OutputCsvPath)

if (-not (Test-Path -LiteralPath $OutputCsvPath)) {
    throw "OpenFace CSV was not created."
}

if (-not $KeepContainerWorkDir) {
    $CleanupCommand = "rm -rf $(Quote-Sh $ContainerRunDir)"
    Invoke-Native -FilePath "docker" -Arguments @("exec", $ContainerRef, "sh", "-lc", $CleanupCommand)
}

Write-Output "created"
