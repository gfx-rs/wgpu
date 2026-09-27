# Appends one line of host performance counters to a CSV every few seconds
# until the sample budget runs out or the job ends and the runner kills us.
#
# Windows CI intermittently stops making progress for a minute or more at a
# time, which kills whichever tests happen to be running. These samples let a
# timeout be correlated with CPU, memory or disk pressure on the host.
# https://github.com/gfx-rs/wgpu/issues/9248
param(
    [Parameter(Mandatory = $true)] [string] $Output,
    [int] $IntervalSeconds = 5,
    [int] $Samples = 400
)

$ErrorActionPreference = "Continue"

$counters = @(
    "\Processor(_Total)\% Processor Time"
    "\Memory\Available MBytes"
    "\Memory\Pages/sec"
    "\PhysicalDisk(_Total)\Current Disk Queue Length"
    "\System\Processes"
)

Set-Content -Path $Output -Encoding ascii `
    -Value "time,cpu_pct,avail_mb,pages_per_sec,disk_queue,processes"

for ($i = 0; $i -lt $Samples; $i++) {
    $now = (Get-Date).ToUniversalTime().ToString("o")
    try {
        $values = (Get-Counter -Counter $counters -ErrorAction Stop).CounterSamples |
            ForEach-Object { [math]::Round($_.CookedValue, 1) }
        Add-Content -Path $Output -Value "$now,$($values -join ',')"
    } catch {
        Add-Content -Path $Output -Value "$now,,,,,"
    }
    Start-Sleep -Seconds $IntervalSeconds
}
