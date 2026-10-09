# Decides whether a failed Windows test job should be retried on another host:
# either the sampler judged the host unsuitable, or the tests failed only by
# timing out. https://github.com/gfx-rs/wgpu/issues/9248
param(
    [Parameter(Mandatory = $true)] [string] $OutputDir,
    [Parameter(Mandatory = $true)] [string] $TestsOutcome,
    [switch] $RetryAllowed
)

$ErrorActionPreference = "Continue"

$verdictFile = Join-Path $OutputDir "host-verdict.txt"
$jobStart = (Get-Item (Join-Path $OutputDir "host-info.txt") -ErrorAction SilentlyContinue).LastWriteTimeUtc

$reason = $null
if (Test-Path $verdictFile) {
    Get-Content $verdictFile
    $reason = "Unsuitable host: $(Get-Content $verdictFile -First 1)"
} elseif ($TestsOutcome -eq "failure") {
    # The target directory is cached, so ignore a junit.xml left by an earlier job.
    $junit = @("target/llvm-cov-target/nextest/default/junit.xml", "target/nextest/default/junit.xml") |
        Where-Object { (Test-Path $_) -and (Get-Item $_).LastWriteTimeUtc -gt $jobStart } |
        Select-Object -First 1
    $xml = if ($junit) { try { [xml](Get-Content $junit -Raw) } catch { $null } }
    if ($xml) {
        $failures = @($xml.SelectNodes("//testcase/failure")) + @($xml.SelectNodes("//testcase/error"))
        $timeouts = @($failures | Where-Object { $_.type -eq "test timeout" }).Count
        if ($timeouts -gt 0 -and $timeouts -eq $failures.Count) {
            $reason = "$timeouts test(s) timed out and nothing else failed."
        }
    }
}

if (-not $reason) {
    exit 0
}
if ($TestsOutcome -eq "success") {
    Write-Output "::warning title=Unsuitable Windows CI host::$reason The tests passed anyway."
} elseif ($RetryAllowed -and $TestsOutcome -eq "failure") {
    Write-Output "::error title=Unsuitable Windows CI host (retrying)::$reason"
    Add-Content -Path $env:GITHUB_OUTPUT -Value "retry=true"
    Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value "Retrying on another host. $reason"
} else {
    Write-Output "::error title=Unsuitable Windows CI host (not retrying)::$reason"
}
