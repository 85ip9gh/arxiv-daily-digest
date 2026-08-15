<#
.SYNOPSIS
    Registers (or removes) the 07:00 arXiv digest scheduled task.

.EXAMPLE
    .\install_nightly.ps1
    .\install_nightly.ps1 -At 06:30
    .\install_nightly.ps1 -Remove
#>
param(
    [string]$TaskName = "ArxivDailyDigest",
    [string]$At = "07:00",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

if ($Remove) {
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        Write-Host "Removed scheduled task '$TaskName'."
    } catch {
        Write-Host "No scheduled task named '$TaskName'."
    }
    exit 0
}

$repo = $PSScriptRoot
$python = (Get-Command python).Source
$log = Join-Path $repo "digests\nightly.log"

New-Item -ItemType Directory -Force -Path (Join-Path $repo "digests") | Out-Null

# cmd /c is here for the redirect: Task Scheduler runs one executable and has
# nowhere to put stdout or stderr on its own, and an unattended job with no log
# is indistinguishable from a job that never ran.
$command = "`"$python`" -m arxiv_digest >> `"$log`" 2>&1"
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c $command" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Daily arXiv AI digest" -Force | Out-Null

Write-Host "Registered '$TaskName' daily at $At."
Write-Host "Log: $log"
Write-Host "Run it now with: Start-ScheduledTask -TaskName $TaskName"
