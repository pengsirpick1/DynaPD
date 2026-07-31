param(
    [int]$Seed = 0,
    [string]$Budgets = "0.30",
    [string]$ParetoBudgets = "",
    [ValidateSet("both", "pseudo", "true")]
    [string]$Mode = "both",
    [string]$Device = "auto",
    [string]$CondaEnv = "llm",
    [string]$PythonExe = "",
    [switch]$UseCondaRun,
    [string]$RunPrefix = "dmmpv3_cw_label_guidance_full",
    [string]$DataRoot = "",
    [string]$OutputDir = "",
    [switch]$SkipFixedAttackEval,
    [switch]$SkipAttackEval,
    [switch]$ForceRetrainAttackers,
    [switch]$DryRun,
    [switch]$NoProgress,
    [string[]]$ExtraTrainArgs = @(),
    [string[]]$ExtraAttackArgs = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($ParetoBudgets)) {
    $ParetoBudgets = $Budgets
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputRoot = Join-Path $ProjectRoot "results"
} else {
    $OutputRoot = [System.IO.Path]::GetFullPath($OutputDir)
}

function Resolve-PythonExe {
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        $candidate = [System.IO.Path]::GetFullPath($PythonExe)
        if (-not (Test-Path $candidate)) {
            throw "PythonExe does not exist: $candidate"
        }
        return $candidate
    }
    $candidates = @(
        (Join-Path $env:USERPROFILE "Miniconda3\envs\$CondaEnv\python.exe"),
        "D:\Miniconda3\envs\$CondaEnv\python.exe",
        "C:\ProgramData\Miniconda3\envs\$CondaEnv\python.exe",
        "C:\ProgramData\Anaconda3\envs\$CondaEnv\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    throw "Cannot find python.exe for env '$CondaEnv'. Pass -PythonExe D:\Miniconda3\envs\$CondaEnv\python.exe, or explicitly use -UseCondaRun."
}

$ResolvedPythonExe = ""
if (-not $UseCondaRun) {
    $ResolvedPythonExe = Resolve-PythonExe
    Write-Host "Using direct Python executable: $ResolvedPythonExe"
}

function Invoke-ExperimentPython {
    param(
        [string]$Label,
        [string[]]$PythonArgs
    )
    Write-Host ""
    Write-Host "========== $Label =========="
    Write-Host ("Started: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    if ($UseCondaRun) {
        Write-Host "conda run --no-capture-output -n $CondaEnv python -u $($PythonArgs -join ' ')"
        if (-not $DryRun) {
            $env:PYTHONUNBUFFERED = "1"
            $env:PYTHONIOENCODING = "utf-8"
            & conda run --no-capture-output -n $CondaEnv python -u @PythonArgs
            $ExitCode = $LASTEXITCODE
        } else {
            $ExitCode = 0
        }
    } else {
        Write-Host "$ResolvedPythonExe -u $($PythonArgs -join ' ')"
        if (-not $DryRun) {
            $env:PYTHONUNBUFFERED = "1"
            $env:PYTHONIOENCODING = "utf-8"
            & $ResolvedPythonExe -u @PythonArgs
            $ExitCode = $LASTEXITCODE
        } else {
            $ExitCode = 0
        }
    }
    if ($ExitCode -ne 0) {
        throw "$Label failed with exit code $ExitCode"
    }
    Write-Host ("Finished: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
}

function Get-BudgetTag {
    param([string]$Value)
    $tag = $Value.Trim() -replace "[^0-9A-Za-z]+", "_"
    return $tag.Trim("_")
}

function Test-AllPaths {
    param([object[]]$Paths)
    foreach ($path in $Paths) {
        if (-not (Test-Path -LiteralPath ([string]$path))) {
            return $false
        }
    }
    return $true
}

function Get-DefenseResumeStages {
    param([string]$RunDir)

    if (-not (Test-Path -LiteralPath $RunDir)) {
        return @("all")
    }
    $existingItems = @(Get-ChildItem -LiteralPath $RunDir -Force -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($existingItems.Count -eq 0) {
        return @("all")
    }

    $stage1Paths = @(
        (Join-Path $RunDir "run_config.json"),
        (Join-Path $RunDir "split_indices.json"),
        (Join-Path $RunDir "stage1_executable_condition\candidate_scorer_checkpoint.pt"),
        (Join-Path $RunDir "stage1_executable_condition\strong_surrogate_ensemble.pt"),
        (Join-Path $RunDir "stage1_executable_condition\candidate_metrics.json")
    )
    $stage2Paths = $stage1Paths + @(
        (Join-Path $RunDir "stage2_user_diffusion\encoder_checkpoint.pt"),
        (Join-Path $RunDir "stage2_user_diffusion\diffusion_guided_checkpoint.pt"),
        (Join-Path $RunDir "stage2_user_diffusion\stage2_metrics.json"),
        (Join-Path $RunDir "stage2_user_diffusion\user_profiles")
    )
    $stage3Paths = $stage2Paths + @(
        (Join-Path $RunDir "stage3_guided_refinement\stage3_metrics.json"),
        (Join-Path $RunDir "stage3_guided_refinement\selected_policy.json")
    )

    if (Test-AllPaths -Paths $stage3Paths) {
        return @()
    }
    if (Test-AllPaths -Paths $stage2Paths) {
        return @("3")
    }
    if (Test-AllPaths -Paths $stage1Paths) {
        return @("2", "3")
    }

    throw "Existing run directory is not recoverable because Stage 1 artifacts are incomplete: $RunDir"
}

$budgetTag = Get-BudgetTag $Budgets
$LogRoot = Join-Path $ProjectRoot "logs"
$LogStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$TranscriptPath = Join-Path $LogRoot "${RunPrefix}_seed${Seed}_b${budgetTag}_label_guidance_${LogStamp}.log"
$modesToRun = @()
if ($Mode -eq "both" -or $Mode -eq "pseudo") {
    $modesToRun += "pseudo"
}
if ($Mode -eq "both" -or $Mode -eq "true") {
    $modesToRun += "true"
}

$completedRuns = @()

$manifest = [ordered]@{
    run_prefix = $RunPrefix
    seed = $Seed
    budgets = $Budgets
    pareto_budgets = $ParetoBudgets
    modes = $modesToRun
    output_root = $OutputRoot
    transcript = $TranscriptPath
    reconnect_resume = $true
    dry_run = [bool]$DryRun
}
$manifestPath = Join-Path $OutputRoot "${RunPrefix}_seed${Seed}_b${budgetTag}_label_guidance_invocation.json"
$manifestJson = $manifest | ConvertTo-Json -Depth 8
Write-Host "Invocation manifest: $manifestPath"
Write-Host "Realtime transcript: $TranscriptPath"
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
    Set-Content -Path $manifestPath -Value $manifestJson -Encoding UTF8
}

$TranscriptStarted = $false
try {
    if (-not $DryRun) {
        try {
            Start-Transcript -Path $TranscriptPath -Append | Out-Null
            $TranscriptStarted = $true
        } catch {
            Write-Warning "Could not start transcript: $($_.Exception.Message)"
        }
    }

    foreach ($guidanceMode in $modesToRun) {
        $suffix = if ($guidanceMode -eq "pseudo") { "pseudo_label_free" } else { "true_label_oracle" }
        $runName = "${RunPrefix}_seed${Seed}_b${budgetTag}_${suffix}"
        $runDir = Join-Path $OutputRoot $runName

        $trainArgs = @(
            "scripts\train_defense.py",
            "--stage", "all",
            "--run_name", $runName,
            "--seed", [string]$Seed,
            "--budgets", $Budgets,
            "--pareto_budgets", $ParetoBudgets,
            "--guidance_label_mode", $guidanceMode,
            "--device", $Device
        )
        if (-not [string]::IsNullOrWhiteSpace($DataRoot)) {
            $trainArgs += @("--data_root", ([System.IO.Path]::GetFullPath($DataRoot)))
        }
        if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
            $trainArgs += @("--output_dir", $OutputRoot)
        }
        if ($NoProgress) {
            $trainArgs += "--no-progress"
        }
        if ($ExtraTrainArgs.Count -gt 0) {
            $trainArgs += $ExtraTrainArgs
        }

        $stagesToRun = @(Get-DefenseResumeStages -RunDir $runDir)
        if ($stagesToRun.Count -eq 0) {
            Write-Host ""
            Write-Host "========== Reusing completed defense run=$runName =========="
            Write-Host "Found Stage 3 artifacts in $runDir"
        } else {
            foreach ($stageToRun in $stagesToRun) {
                $stageTrainArgs = [string[]]$trainArgs.Clone()
                $stageIndex = [array]::IndexOf($stageTrainArgs, "--stage")
                if ($stageIndex -lt 0) {
                    throw "Internal error: train args are missing --stage"
                }
                $stageTrainArgs[$stageIndex + 1] = $stageToRun
                Invoke-ExperimentPython "Train DMMPv3 guidance_label_mode=$guidanceMode stage=$stageToRun run=$runName" $stageTrainArgs
            }
        }
        $completedRuns += $runDir

        if (-not ($SkipFixedAttackEval -or $SkipAttackEval)) {
            $fixedArgs = @(
                "scripts\run_attack_eval.py",
                "--run_dir", $runDir,
                "--seed", [string]$Seed,
                "--device", $Device,
                "--attackers", "fixed_df,fixed_rf",
                "--adaptive_protocol", "fixed"
            )
            if (-not [string]::IsNullOrWhiteSpace($DataRoot)) {
                $dataRootFull = [System.IO.Path]::GetFullPath($DataRoot)
                $fixedArgs += @("--data_root", $dataRootFull)
            }
            if ($ForceRetrainAttackers) {
                $fixedArgs += "--force_retrain"
            }
            if ($NoProgress) {
                $fixedArgs += "--no-progress"
            }
            if ($ExtraAttackArgs.Count -gt 0) {
                $fixedArgs += $ExtraAttackArgs
            }

            Invoke-ExperimentPython "Evaluate fixed DF/RF run=$runName" $fixedArgs
        }

        if (-not $DryRun -and (Test-Path -LiteralPath $runDir)) {
            Set-Content -Path (Join-Path $runDir "label_guidance_invocation.json") -Value $manifestJson -Encoding UTF8
        }
    }
} finally {
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
}

Write-Host ""
Write-Host "========== Completed full label-guidance experiment =========="
foreach ($runDir in $completedRuns) {
    Write-Host $runDir
}
Write-Host "Realtime transcript: $TranscriptPath"
