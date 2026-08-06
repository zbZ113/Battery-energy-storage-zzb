[CmdletBinding()]
param(
    [string]$RuntimeRoot = 'D:\QuanxinRuntime'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

if (-not (Test-IsAdministrator)) {
    $arguments = @(
        '-NoProfile'
        '-ExecutionPolicy', 'Bypass'
        '-File', ('"{0}"' -f $PSCommandPath)
        '-RuntimeRoot', ('"{0}"' -f $RuntimeRoot)
    )
    $elevation = @{
        FilePath = 'powershell.exe'
        Verb = 'RunAs'
        ArgumentList = $arguments
        Wait = $true
        PassThru = $true
    }
    $elevated = Start-Process @elevation
    exit $elevated.ExitCode
}

$runtimePath = [IO.Path]::GetFullPath($RuntimeRoot)
if (-not $runtimePath.StartsWith('D:\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'RuntimeRoot must remain on D drive'
}

$drive = Get-PSDrive -Name D
if ($drive.Free -lt 100GB) {
    throw 'D drive requires at least 100 GiB free'
}

$installRoot = Join-Path $runtimePath 'docker-desktop'
$wslDataRoot = Join-Path $runtimePath 'docker-wsl'
$windowsDataRoot = Join-Path $runtimePath 'docker-windows'
$downloadRoot = Join-Path $runtimePath 'downloads'
foreach ($path in @(
    $runtimePath,
    $installRoot,
    $wslDataRoot,
    $windowsDataRoot,
    $downloadRoot
)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

$restartRequired = $false
foreach ($featureName in @(
    'Microsoft-Windows-Subsystem-Linux',
    'VirtualMachinePlatform'
)) {
    $feature = Get-WindowsOptionalFeature -Online -FeatureName $featureName
    if ($feature.State -ne 'Enabled') {
        $result = Enable-WindowsOptionalFeature -Online `
            -FeatureName $featureName -All -NoRestart
        $restartRequired = $restartRequired -or $result.RestartNeeded
    }
}

if ($restartRequired) {
    Write-Output 'REBOOT_REQUIRED'
    exit 3010
}

& wsl.exe --update --web-download
if ($LASTEXITCODE -ne 0) {
    Write-Output (
        'WSL web update unavailable; installing the signed inbox-WSL kernel update'
    )
    $kernelInstaller = Join-Path $downloadRoot 'wsl_update_x64.msi'
    $kernelInstallerUrl = (
        'https://wslstorestorage.blob.core.windows.net/wslblob/' +
        'wsl_update_x64.msi'
    )
    if (-not (Test-Path -LiteralPath $kernelInstaller -PathType Leaf)) {
        Invoke-WebRequest `
            -Uri $kernelInstallerUrl `
            -OutFile $kernelInstaller `
            -UseBasicParsing
    }

    $kernelSignature = Get-AuthenticodeSignature -LiteralPath $kernelInstaller
    $isMicrosoftSigner = (
        $null -ne $kernelSignature.SignerCertificate -and
        $kernelSignature.SignerCertificate.Subject -match `
            'O=Microsoft Corporation'
    )
    if (
        $kernelSignature.Status -ne [Management.Automation.SignatureStatus]::Valid -or
        -not $isMicrosoftSigner
    ) {
        throw 'WSL2 kernel update does not have a valid Microsoft Corporation signature'
    }

    $kernelInstallation = Start-Process `
        -FilePath 'msiexec.exe' `
        -ArgumentList @('/i', ('"{0}"' -f $kernelInstaller), '/qn', '/norestart') `
        -Wait `
        -PassThru
    if ($kernelInstallation.ExitCode -eq 3010) {
        Write-Output 'REBOOT_REQUIRED'
        exit 3010
    }
    if ($kernelInstallation.ExitCode -ne 0) {
        throw (
            'WSL2 kernel update failed with exit code {0}' -f `
                $kernelInstallation.ExitCode
        )
    }

    & wsl.exe --update --web-download
    if ($LASTEXITCODE -ne 0) {
        throw "WSL update failed after kernel installation with exit code $LASTEXITCODE"
    }
}
& wsl.exe --status
if ($LASTEXITCODE -ne 0) {
    throw "WSL status check failed with exit code $LASTEXITCODE"
}
& wsl.exe --set-default-version 2
if ($LASTEXITCODE -ne 0) {
    throw "WSL2 default version setup failed with exit code $LASTEXITCODE"
}

$installer = Join-Path $downloadRoot 'Docker Desktop Installer.exe'
$installerUrl = (
    'https://desktop.docker.com/win/main/amd64/' +
    'Docker%20Desktop%20Installer.exe'
)
if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    Invoke-WebRequest -Uri $installerUrl -OutFile $installer -UseBasicParsing
}

$installArguments = @(
    'install'
    '--quiet'
    '--accept-license'
    '--backend=wsl-2'
    ('--installation-dir={0}' -f $installRoot)
    ('--wsl-default-data-root={0}' -f $wslDataRoot)
    ('--windows-containers-default-data-root={0}' -f $windowsDataRoot)
)
$installerProcess = @{
    FilePath = $installer
    ArgumentList = $installArguments
    Wait = $true
    PassThru = $true
}
$installation = Start-Process @installerProcess
if ($installation.ExitCode -eq 3010) {
    Write-Output 'REBOOT_REQUIRED'
    exit 3010
}
if ($installation.ExitCode -ne 0) {
    throw "Docker Desktop installer failed with exit code $($installation.ExitCode)"
}

$dockerDesktop = Join-Path $installRoot 'Docker Desktop.exe'
$dockerCli = Join-Path $installRoot 'resources\bin\docker.exe'
if (-not (Test-Path -LiteralPath $dockerDesktop -PathType Leaf)) {
    throw 'Docker Desktop executable is missing after installation'
}
if (-not (Test-Path -LiteralPath $dockerCli -PathType Leaf)) {
    throw 'Docker CLI is missing after installation'
}

Start-Process -FilePath $dockerDesktop -WindowStyle Hidden | Out-Null
$deadline = [DateTime]::UtcNow.AddMinutes(5)
do {
    & $dockerCli info *> $null
    if ($LASTEXITCODE -eq 0) {
        break
    }
    Start-Sleep -Seconds 2
} while ([DateTime]::UtcNow -lt $deadline)

if ($LASTEXITCODE -ne 0) {
    throw 'Docker Desktop did not become ready within five minutes'
}

Write-Output 'Verifying docker compose version'
& $dockerCli version
if ($LASTEXITCODE -ne 0) {
    throw 'docker version failed'
}
& $dockerCli compose version
if ($LASTEXITCODE -ne 0) {
    throw 'docker compose version failed'
}

Write-Output ('DOCKER_INSTALL_ROOT={0}' -f $installRoot)
Write-Output ('DOCKER_WSL_DATA_ROOT={0}' -f $wslDataRoot)
Write-Output ('DOCKER_WINDOWS_DATA_ROOT={0}' -f $windowsDataRoot)
