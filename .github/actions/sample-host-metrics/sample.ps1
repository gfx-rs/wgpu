# Appends host performance counters to CSVs in $OutputDir every few seconds
# until the time budget runs out or the job ends and the runner kills us.
#
# Windows CI intermittently lands on hosts that stop making progress for a
# minute or more at a time, which kills whichever tests happen to be running.
# These samples let a timeout be correlated with CPU, memory or disk pressure
# on the host. https://github.com/gfx-rs/wgpu/issues/9248
#
# While the watchdog arm file exists, the sampler also judges the host. On
# those hosts this loop slows down along with everything else: healthy runners
# stay near 6.1s per iteration, and every stalling runner seen so far averaged
# more than $Threshold seconds over $Window iterations while still building.
# The extra counters are excluded from the measured period so that it stays
# comparable with the data the threshold was chosen from.
param(
    [Parameter(Mandatory = $true)] [string] $OutputDir,
    [int] $IntervalSeconds = 5,
    [int] $MaxMinutes = 45,
    [switch] $AbortOnBadHost,
    [int] $Window = 10,
    [double] $Threshold = 7.0
)

$ErrorActionPreference = "Continue"

$metricsFile = Join-Path $OutputDir "host-metrics.csv"
$countersFile = Join-Path $OutputDir "host-counters.csv"
$processesFile = Join-Path $OutputDir "host-processes.csv"
$verdictFile = Join-Path $OutputDir "host-verdict.txt"
$armFile = Join-Path $OutputDir "watchdog-armed"

$baseCounters = @(
    "\Processor(_Total)\% Processor Time"
    "\Memory\Available MBytes"
    "\Memory\Pages/sec"
    "\PhysicalDisk(_Total)\Current Disk Queue Length"
    "\System\Processes"
)

$extraCounters = @(
    "\PhysicalDisk(*)\% Disk Time"
    "\PhysicalDisk(*)\% Idle Time"
    "\PhysicalDisk(*)\Disk Bytes/sec"
    "\PhysicalDisk(*)\Disk Read Bytes/sec"
    "\PhysicalDisk(*)\Disk Write Bytes/sec"
    "\PhysicalDisk(*)\Disk Transfers/sec"
    "\PhysicalDisk(*)\Avg. Disk sec/Transfer"
    "\PhysicalDisk(*)\Avg. Disk sec/Read"
    "\PhysicalDisk(*)\Avg. Disk sec/Write"
    "\PhysicalDisk(*)\Current Disk Queue Length"
    "\Processor(_Total)\% User Time"
    "\Processor(_Total)\% Privileged Time"
    "\Processor(_Total)\% Interrupt Time"
    "\Processor(_Total)\% DPC Time"
    "\Processor Information(_Total)\% Processor Performance"
    "\Processor Information(_Total)\Processor Frequency"
    "\System\Processor Queue Length"
    "\System\Context Switches/sec"
    "\System\Threads"
    "\Memory\Committed Bytes"
    "\Memory\Cache Bytes"
    "\Memory\Modified Page List Bytes"
    "\Memory\Page Faults/sec"
    "\Memory\Page Reads/sec"
    "\Memory\Page Writes/sec"
    "\Paging File(_Total)\% Usage"
)

$processCounters = @(
    "\Process(*)\% Processor Time"
    "\Process(*)\IO Data Bytes/sec"
    "\Process(*)\ID Process"
)

Set-Content -Path $metricsFile -Encoding ascii `
    -Value "time,cpu_pct,avail_mb,pages_per_sec,disk_queue,processes,loop_s,extra_s,armed"
Set-Content -Path $countersFile -Encoding ascii -Value "time,counter,value"
Set-Content -Path $processesFile -Encoding ascii -Value "time,kind,rank,name,pid,value"

function Get-TopProcesses($samples, $now) {
    $byInstance = @{}
    foreach ($s in $samples) {
        if ($s.InstanceName -in "_total", "idle") { continue }
        if (-not $byInstance.ContainsKey($s.InstanceName)) {
            $byInstance[$s.InstanceName] = @{ name = $s.InstanceName }
        }
        $byInstance[$s.InstanceName][$s.Path.Split("\")[-1]] = $s.CookedValue
    }
    $rows = @()
    foreach ($kind in @(@("cpu", "% processor time"), @("io", "io data bytes/sec"))) {
        $rank = 0
        $byInstance.Values | Where-Object { $_[$kind[1]] -gt 0 } | Sort-Object { $_[$kind[1]] } -Descending |
            Select-Object -First 5 |
            ForEach-Object {
                $rank++
                $rows += "$now,$($kind[0]),$rank,$($_.name),$($_["id process"]),$([math]::Round($_[$kind[1]], 1))"
            }
    }
    $rows
}

$clock = [Diagnostics.Stopwatch]::StartNew()
$recent = [Collections.Generic.Queue[double]]::new()
$fired = $false
$prevStart = $null
$prevExtra = 0.0

while ($clock.Elapsed.TotalMinutes -lt $MaxMinutes) {
    $start = $clock.Elapsed.TotalSeconds
    $loop = if ($null -ne $prevStart) { $start - $prevStart - $prevExtra } else { $null }
    $now = (Get-Date).ToUniversalTime().ToString("o")

    try {
        $base = ((Get-Counter -Counter $baseCounters -ErrorAction Stop).CounterSamples |
            ForEach-Object { [math]::Round($_.CookedValue, 1) }) -join ","
    } catch {
        $base = ",,,,"
    }

    $extraStart = $clock.Elapsed.TotalSeconds
    $samples = (Get-Counter -Counter ($extraCounters + $processCounters) -ErrorAction SilentlyContinue).CounterSamples
    $counterRows = $samples | Where-Object { $_.Path -notmatch "\\process\(" } | ForEach-Object {
        "$now,$($_.Path -replace '^\\\\[^\\]+', ''),$([math]::Round($_.CookedValue, 4))"
    }
    $processRows = Get-TopProcesses ($samples | Where-Object { $_.Path -match "\\process\(" }) $now
    $prevExtra = $clock.Elapsed.TotalSeconds - $extraStart

    $armed = Test-Path $armFile
    $loopText = if ($null -ne $loop) { [math]::Round($loop, 2) } else { "" }
    Add-Content -Path $metricsFile -Value "$now,$base,$loopText,$([math]::Round($prevExtra, 2)),$([int]$armed)"
    if ($counterRows) { Add-Content -Path $countersFile -Value $counterRows }
    if ($processRows) { Add-Content -Path $processesFile -Value $processRows }

    if ($armed -and $null -ne $loop) {
        $recent.Enqueue($loop)
        while ($recent.Count -gt $Window) { [void]$recent.Dequeue() }
        $mean = ($recent | Measure-Object -Average).Average
        if (-not $fired -and $recent.Count -eq $Window -and $mean -gt $Threshold) {
            $fired = $true
            Set-Content -Path $verdictFile -Value (
                "$now The host metrics sampler averaged $([math]::Round($mean, 2))s per iteration over its " +
                "last $Window iterations (threshold $($Threshold)s; healthy runners stay near 6.1s).")
            if ($AbortOnBadHost) {
                taskkill /F /T /IM cargo.exe 2>&1 | Add-Content -Path $verdictFile
            }
        }
    } else {
        $recent.Clear()
    }

    $prevStart = $start
    Start-Sleep -Seconds $IntervalSeconds
}
