[CmdletBinding()]
param(
    [string]$RuntimeRoot = 'D:\QuanxinRuntime',
    [switch]$SkipImages
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function New-UrlSafeSecret {
    $bytes = [byte[]]::new(48)
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    return [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
}

function Set-RestrictedAcl([string]$Path) {
    $currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    & icacls.exe $Path '/inheritance:r' | Out-Null
    & icacls.exe $Path '/grant:r' "*$currentSid`:(F)" | Out-Null
    & icacls.exe $Path '/grant:r' '*S-1-5-18:(F)' | Out-Null
    & icacls.exe $Path '/grant:r' '*S-1-5-32-544:(F)' | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to restrict ACL for $Path"
    }
}

function Write-Utf8File([string]$Path, [string]$Value) {
    [IO.File]::WriteAllText(
        $Path,
        $Value,
        [Text.UTF8Encoding]::new($false)
    )
}

function Resolve-DockerCli([string]$Root) {
    $command = Get-Command docker.exe -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    $candidate = Join-Path $Root 'docker-desktop\resources\bin\docker.exe'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        return $candidate
    }
    throw 'Docker CLI is unavailable; rerun after Docker Desktop installation'
}

function Resolve-ImageDigest(
    [string]$DockerCli,
    [string]$Image
) {
    & $DockerCli pull $Image | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to pull reviewed image $Image"
    }
    $reference = (& $DockerCli image inspect --format '{{index .RepoDigests 0}}' $Image).Trim()
    if ($LASTEXITCODE -ne 0 -or $reference -notmatch '@(sha256:[0-9a-f]{64})$') {
        throw "Reviewed image has no immutable digest: $Image"
    }
    return $Matches[1]
}

$root = [IO.Path]::GetFullPath($RuntimeRoot)
if (-not $root.StartsWith('D:\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'RuntimeRoot must remain on D drive'
}
$drive = Get-PSDrive -Name D
if ($drive.Free -lt 60GB) {
    throw 'D drive requires at least 60 GiB free before image preparation'
}

$directories = @(
    $root,
    (Join-Path $root 'compose'),
    (Join-Path $root 'runtime\data'),
    (Join-Path $root 'runtime\artifacts'),
    (Join-Path $root 'runtime\policies'),
    (Join-Path $root 'runtime\registry'),
    (Join-Path $root 'runtime\calibration'),
    (Join-Path $root 'runtime\config'),
    (Join-Path $root 'runtime\demo-target'),
    (Join-Path $root 'secrets'),
    (Join-Path $root 'backups'),
    (Join-Path $root 'logs')
)
foreach ($directory in $directories) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

$secretRoot = Join-Path $root 'secrets'
$secretNames = @(
    'postgres_password',
    'database_url',
    'redis_password',
    'redis_acl',
    'redis_url'
)
$existingSecrets = @(
    $secretNames | Where-Object {
        Test-Path -LiteralPath (Join-Path $secretRoot $_) -PathType Leaf
    }
)
if ($existingSecrets.Count -ne 0 -and $existingSecrets.Count -ne $secretNames.Count) {
    throw 'Secret set is partial; repair it manually without regenerating credentials'
}
if ($existingSecrets.Count -eq 0) {
    $postgres = New-UrlSafeSecret
    $redis = New-UrlSafeSecret
    $secretValues = [ordered]@{
        postgres_password = "$postgres`n"
        database_url = "postgresql+psycopg://quanxin:$postgres@postgres:5432/quanxin`n"
        redis_password = "$redis`n"
        redis_acl = "user default off`nuser quanxin on >$redis ~* &* +@all`n"
        redis_url = "redis://quanxin:$redis@redis:6379/0`n"
    }
    foreach ($entry in $secretValues.GetEnumerator()) {
        $path = Join-Path $secretRoot $entry.Key
        Write-Utf8File -Path $path -Value $entry.Value
        Set-RestrictedAcl -Path $path
    }
    $postgres = $null
    $redis = $null
    $secretValues = $null
}

$postgresDigest = ''
$redisDigest = ''
if (-not $SkipImages) {
    $dockerCli = Resolve-DockerCli -Root $root
    $dockerBin = [IO.Path]::GetDirectoryName($dockerCli)
    $env:PATH = "$dockerBin$([IO.Path]::PathSeparator)$env:PATH"
    $postgresDigest = Resolve-ImageDigest -DockerCli $dockerCli `
        -Image 'pgvector/pgvector:0.8.1-pg16'
    $redisDigest = Resolve-ImageDigest -DockerCli $dockerCli `
        -Image 'redis:7.4.2-alpine'
}

$gitRevision = (& git rev-parse --short=12 HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $gitRevision) {
    $gitRevision = 'working'
}
$environmentFile = Join-Path $root 'compose\local.env'
$environment = @(
    "LOCAL_RUNTIME_ROOT=$root",
    "QUANXIN_SECRETS_ROOT=$secretRoot",
    "POSTGRES_IMAGE_DIGEST=$postgresDigest",
    "REDIS_IMAGE_DIGEST=$redisDigest",
    'DEPLOYMENT_REGISTRY_ID=c31f62e68faa66e56b16d21ebdd3067d5dea0c8408bb3ad6baa73e05a42824be',
    'PUBLIC_ORIGIN=http://localhost:8080',
    "LOCAL_IMAGE_TAG=$gitRevision"
) -join "`n"
Write-Utf8File -Path $environmentFile -Value ($environment + "`n")

Write-Output ('RUNTIME_ROOT_READY={0}' -f $root)
Write-Output ('LOCAL_ENV_READY={0}' -f $environmentFile)
Get-ChildItem -LiteralPath $secretRoot -File |
    Select-Object Name, Length
