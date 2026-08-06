[CmdletBinding()]
param(
    [string]$RuntimeRoot = 'D:\QuanxinRuntime',
    [string]$Username = 'admin@quanxin.local'
)

$ErrorActionPreference = 'Stop'

$root = [IO.Path]::GetFullPath($RuntimeRoot)
if (-not $root.StartsWith('D:\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'RuntimeRoot must remain on D drive'
}
$environmentFile = Join-Path $root 'compose\local.env'
if (-not (Test-Path -LiteralPath $environmentFile -PathType Leaf)) {
    throw 'Local environment file is missing'
}

$secure = Read-Host -AsSecureString 'Enter the temporary local ADMIN password'
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$temporaryFile = Join-Path (
    Join-Path $root 'secrets'
) ('.bootstrap_admin_password.{0}' -f [Guid]::NewGuid())
try {
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    [IO.File]::WriteAllText(
        $temporaryFile,
        ($plain + "`n"),
        [Text.UTF8Encoding]::new($false)
    )
    $currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    & icacls.exe $temporaryFile '/inheritance:r' | Out-Null
    & icacls.exe $temporaryFile '/grant:r' "*$currentSid`:(F)" | Out-Null
    & icacls.exe $temporaryFile '/grant:r' '*S-1-5-18:(F)' | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to restrict temporary ADMIN secret ACL'
    }

    $mount = '{0}:/run/secrets/bootstrap_admin_password:ro' -f $temporaryFile
    $arguments = @(
        'compose',
        '--env-file', $environmentFile,
        '-f', 'deploy/local.compose.yaml',
        'run', '--rm', '--no-deps',
        '-e', "QUANXIN_BOOTSTRAP_ADMIN_USERNAME=$Username",
        '-e', 'QUANXIN_BOOTSTRAP_ADMIN_PASSWORD_FILE=/run/secrets/bootstrap_admin_password',
        '-v', $mount,
        'api', 'python', '-m', 'deploy.bootstrap_admin'
    )
    # Runs: python -m deploy.bootstrap_admin inside the local backend image.
    & docker @arguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Local ADMIN bootstrap failed'
    }
}
finally {
    $plain = $null
    $secure = $null
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    if (Test-Path -LiteralPath $temporaryFile) {
        Remove-Item -LiteralPath $temporaryFile -Force
    }
}
