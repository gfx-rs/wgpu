# Prints, and saves to $Output, whatever might distinguish one CI host from
# another. https://github.com/gfx-rs/wgpu/issues/9248
param(
    [Parameter(Mandatory = $true)] [string] $Output
)

$ErrorActionPreference = "Continue"

function Show($title, [scriptblock] $block) {
    "==== $title"
    try {
        & $block 2>&1 | Format-List | Out-String -Width 4096
    } catch {
        "failed: $_"
    }
}

& {
    Show "Computer system" {
        Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, SystemFamily,
            HypervisorPresent, NumberOfProcessors, NumberOfLogicalProcessors, TotalPhysicalMemory
    }
    Show "BIOS" { Get-CimInstance Win32_BIOS | Select-Object Manufacturer, SMBIOSBIOSVersion, Version, ReleaseDate }
    Show "Processor" {
        Get-CimInstance Win32_Processor | Select-Object Name, Manufacturer, Description, NumberOfCores,
            NumberOfLogicalProcessors, MaxClockSpeed, CurrentClockSpeed, L2CacheSize, L3CacheSize
    }
    Show "Operating system" {
        Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, LastBootUpTime,
            TotalVisibleMemorySize, FreePhysicalMemory, SizeStoredInPagingFiles
    }
    Show "Page files" { Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage }
    Show "Physical disks" {
        Get-PhysicalDisk | Select-Object DeviceId, FriendlyName, Model, MediaType, BusType, SpindleSpeed, Size,
            LogicalSectorSize, PhysicalSectorSize, FirmwareVersion, HealthStatus, OperationalStatus
    }
    Show "Disk drives" {
        Get-CimInstance Win32_DiskDrive | Select-Object Index, Model, InterfaceType, MediaType, Size,
            SCSIBus, SCSIPort, SCSITargetId, SCSILogicalUnit
    }
    Show "Partitions" { Get-Partition | Select-Object DiskNumber, PartitionNumber, DriveLetter, Type, Size }
    Show "Volumes" {
        Get-Volume | Select-Object DriveLetter, FileSystemLabel, FileSystem, DriveType, AllocationUnitSize,
            Size, SizeRemaining, HealthStatus
    }
    Show "Dev Drive" { foreach ($d in "C:", "D:") { "$d $(fsutil devdrv query $d 2>&1)" } }
    Show "File system filters" { fltmc filters; fltmc instances }
    Show "Defender status" {
        Get-MpComputerStatus | Select-Object AMRunningMode, AMServiceEnabled, AntivirusEnabled,
            RealTimeProtectionEnabled, OnAccessProtectionEnabled, IoavProtectionEnabled, BehaviorMonitorEnabled,
            IsTamperProtected, AMEngineVersion, AntivirusSignatureLastUpdated, QuickScanStartTime,
            QuickScanEndTime, FullScanStartTime, FullScanEndTime
    }
    Show "Defender preferences" {
        Get-MpPreference | Select-Object DisableRealtimeMonitoring, DisableBehaviorMonitoring, DisableIOAVProtection,
            DisableScriptScanning, DisableArchiveScanning, ExclusionPath, ExclusionProcess, ScanAvgCPULoadFactor,
            ScanScheduleDay, EnableControlledFolderAccess
    }
    Show "Services" {
        Get-Service WinDefend, Sense, WSearch, SysMain, wuauserv, TrustedInstaller, DiagTrack, defragsvc `
            -ErrorAction SilentlyContinue | Select-Object Name, Status, StartType
    }
    Show "Running scheduled tasks" {
        Get-ScheduledTask | Where-Object State -eq Running | Select-Object TaskPath, TaskName
    }
    Show "Power scheme" { powercfg /getactivescheme }
    Show "Azure instance metadata" {
        Invoke-RestMethod -Headers @{ Metadata = "true" } -TimeoutSec 3 `
            -Uri "http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01" |
            Select-Object location, zone, vmSize, platformFaultDomain, platformUpdateDomain, sku, version
    }
    Show "Top processes by CPU time" {
        Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 Name, Id, CPU, WorkingSet64
    }
} | Tee-Object -FilePath $Output
