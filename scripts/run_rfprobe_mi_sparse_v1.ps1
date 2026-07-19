param(
    [string]$RunName = "",
    [string]$DataRoot = "D:\learning\TOR\datasets\CW",
    [string]$OutputDir = "results",
    [int]$Seed = 0,
    [string]$CondaEnv = "llm",
    [double]$Budget = 0.30,
    [ValidateSet("1", "2", "3", "all")]
    [string]$DefenseStage = "all",
    [int]$DiffusionTrainSteps = 30000,
    [int]$EncoderEpochs = 10,
    [int]$SurrogateEpochs = 10,
    [int]$AttackEpochs = 10,
    [int]$BatchSize = 128,
    [int]$AttackBatchSize = 256,
    [int]$DeploymentRepeats = 3,
    [int]$ParetoSamples = 4096,
    [string]$RefineKeepRatios = "1.0,0.95,0.90",
    [int]$Stage3FixedProbeSamples = 1024,
    [int]$Stage3FixedProbeTrainSamples = 8000,
    [int]$Stage3FixedProbeValSamples = 2500,
    [int]$Stage3FixedProbeEpochs = 5,
    [double]$DirectionCorrectionStrength = 0.50,
    [double]$MinIncomingDummyShare = 0.20,
    [double]$MinTamIncomingL1Shift = 0.12,
    [int]$LogEvery = 250,
    [switch]$SkipDefense,
    [switch]$SkipFixedRfAttack,
    [switch]$ForceRetrainAttack,
    [switch]$SkipSummary,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not $RunName.Trim()) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunName = "dmmpv3_rfprobe_mi_sparse_v1_fullcw_seed${Seed}_${stamp}"
}

$RunDir = Join-Path $OutputDir $RunName
$BudgetText = [string]::Format("{0:0.00}", $Budget)
$LogRoot = Join-Path $ProjectRoot "logs"
$LogStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$TranscriptPath = Join-Path $LogRoot "${RunName}_rfprobe_mi_sparse_v1_${LogStamp}.log"

function Enable-RealtimeCondaCommand {
    param([string[]]$Command)
    $RealtimeCommand = New-Object System.Collections.Generic.List[string]
    $RealtimeCommand.AddRange([string[]]$Command)
    if ($RealtimeCommand.Count -gt 0 -and $RealtimeCommand[0] -eq "run" -and -not $RealtimeCommand.Contains("--no-capture-output")) {
        $RealtimeCommand.Insert(1, "--no-capture-output")
    }
    $PythonIndex = $RealtimeCommand.IndexOf("python")
    if ($PythonIndex -ge 0 -and ($PythonIndex -eq ($RealtimeCommand.Count - 1) -or $RealtimeCommand[$PythonIndex + 1] -ne "-u")) {
        $RealtimeCommand.Insert($PythonIndex + 1, "-u")
    }
    return $RealtimeCommand.ToArray()
}

function Invoke-ExperimentStep {
    param(
        [string]$Name,
        [string[]]$Command
    )
    $RealtimeCommand = Enable-RealtimeCondaCommand -Command $Command
    Write-Host ""
    Write-Host "========== $Name =========="
    Write-Host ("Started: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ("conda " + ($RealtimeCommand -join " "))
    if (-not $DryRun) {
        $env:PYTHONUNBUFFERED = "1"
        $env:PYTHONIOENCODING = "utf-8"
        & conda @RealtimeCommand
        $ExitCode = $LASTEXITCODE
        if ($ExitCode -ne 0) {
            throw "Experiment step failed with exit code $ExitCode`: $Name"
        }
    }
    Write-Host ("Finished: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
}

$defenseCmd = @(
    "run", "-n", $CondaEnv, "python", "scripts\train_defense.py",
    "--version", "v3",
    "--stage", $DefenseStage,
    "--data_root", $DataRoot,
    "--output_dir", $OutputDir,
    "--run_name", $RunName,
    "--seed", ([string]$Seed),
    "--max_samples", "0",
    "--max_classes", "0",
    "--max_generation_traces", "0",
    "--generation_split", "test",
    "--val_ratio", "0.10",
    "--test_ratio", "0.10",
    "--budgets", $BudgetText,
    "--pareto_budgets", $BudgetText,
    "--pareto_samples", ([string]$ParetoSamples),
    "--refine_keep_ratios", $RefineKeepRatios,
    "--profile_combination_mode", "legacy_pool",
    "--active_pair_count", "10",
    "--active_triple_count", "5",
    "--pair_probability", "0.67",
    "--dirichlet_alpha", "1.0",
    "--v1_mode_pool", "legacy_direct",
    "--v1_mode_prior_weight", "0.65",
    "--condition_preference_map",
    "--condition_selected_mask",
    "--condition_preference_weights",
    "--no-condition_profile_mask",
    "--render_coordinate", "tam_obfuscation",
    "--tam_obfuscation_strategy", "hybrid_clustered",
    "--tam_slot_jitter", "0.03",
    "--tam_cluster_ratio", "0.70",
    "--tam_local_run_max", "8",
    "--tam_preserve_real_timestamps",
    "--surrogate_train_samples", "0",
    "--surrogate_val_samples", "0",
    "--surrogate_epochs", ([string]$SurrogateEpochs),
    "--surrogate_patience", "3",
    "--surrogate_df_architecture", "project",
    "--guidance_attackers", "rf",
    "--surrogate_df_weight", "0.0",
    "--surrogate_rf_weight", "1.0",
    "--surrogate_robust_weight", "0.60",
    "--encoder_train_samples", "1000000",
    "--encoder_epochs", ([string]$EncoderEpochs),
    "--diffusion_train_steps", ([string]$DiffusionTrainSteps),
    "--batch_size", ([string]$BatchSize),
    "--probe_samples", "1000000",
    "--probe_exact_samples", "256",
    "--probe_attacker", "rf",
    "--view_profile_samples", "0",
    "--candidate_mode", "executable",
    "--candidate_topk", "80",
    "--candidate_soft_topk",
    "--candidate_temperature", "0.15",
    "--policy_generator", "diffusion",
    "--sampling_steps", "20",
    "--guidance_weight", "0.12",
    "--guidance_last_steps", "6",
    "--guidance_train_steps", ([string]$DiffusionTrainSteps),
    "--defense_hard_weight", "1.0",
    "--defense_soft_objective_scale", "0.05",
    "--defense_soft_utility_weight", "0.05",
    "--defense_risk_tolerance", "0.0",
    "--prior_leak_weight", "1.70",
    "--prior_preference_weight", "0.10",
    "--prior_noise_std", "0.0",
    "--preference_weight", "0.01",
    "--preference_attack_gate",
    "--preference_attack_gate_margin", "0.02",
    "--constraint_weight", "0.02",
    "--profile_weight", "0.0",
    "--refine_method", "continuous",
    "--refine_steps", "6",
    "--direction_target", "incoming",
    "--direction_correction_strength", ([string]$DirectionCorrectionStrength),
    "--min_incoming_dummy_share", ([string]$MinIncomingDummyShare),
    "--deployment_repeats", ([string]$DeploymentRepeats),
    "--stage3_repeats", "1",
    "--stage3_fixed_probe_samples", ([string]$Stage3FixedProbeSamples),
    "--stage3_fixed_probe_train_samples", ([string]$Stage3FixedProbeTrainSamples),
    "--stage3_fixed_probe_val_samples", ([string]$Stage3FixedProbeValSamples),
    "--stage3_fixed_probe_epochs", ([string]$Stage3FixedProbeEpochs),
    "--stage3_fixed_probe_attackers", "rf",
    "--stage3_fixed_probe_weight", "1.50",
    "--stage3_fixed_probe_min_clean_accuracy", "0.70",
    "--stage3_max_label_free_attack_pressure", "0.45",
    "--stage3_max_attack_accuracy", "0.40",
    "--stage3_max_rendered_rf_accuracy", "0.40",
    "--stage3_max_reliable_fixed_probe_accuracy", "0.40",
    "--stage3_min_dummy_incoming_share", ([string]$MinIncomingDummyShare),
    "--stage3_min_tam_incoming_l1_shift", ([string]$MinTamIncomingL1Shift),
    "--stage3_incoming_metric_weight", "0.75",
    "--no-stage3_require_quality_gate",
    "--preserve_variable_length_traces",
    "--progress",
    "--log_every", ([string]$LogEvery)
)

$attackCommon = @(
    "--run_dir", $RunDir,
    "--data_root", $DataRoot,
    "--seed", ([string]$Seed),
    "--device", "auto",
    "--max_train_traces", "0",
    "--max_val_traces", "0",
    "--max_test_traces", "0",
    "--clean_df_epochs", ([string]$AttackEpochs),
    "--clean_df_patience", "3",
    "--adaptive_epochs", "0",
    "--adaptive_patience", "1",
    "--adaptive_init", "checkpoint",
    "--df_batch_size", ([string]$AttackBatchSize),
    "--df_architecture", "project",
    "--rf_tam_num_slots", "1800",
    "--max_load_time", "80.0",
    "--no-attack_require_quality_gate",
    "--progress",
    "--log_every", ([string]$LogEvery)
)

$fixedRfAttackCmd = @(
    "run", "-n", $CondaEnv, "python", "scripts\run_attack_eval.py",
    "--attackers", "fixed_rf",
    "--adaptive_protocol", "fixed"
) + $attackCommon
if ($ForceRetrainAttack) {
    $fixedRfAttackCmd += "--force_retrain"
}

$summaryCmd = @(
    "run", "-n", $CondaEnv, "python", "scripts\summarize_full_legacy_fused_diffusion.py",
    "--run_dir", $RunDir,
    "--protocols", "fixed"
)

$manifest = [ordered]@{
    run_name = $RunName
    run_dir = $RunDir
    method = "dmmpv3_rfprobe_mi_sparse_v1"
    budget = $Budget
    full_cw = $true
    rf_only_attack = $true
    realtime_output = $true
    conda_no_capture_output = $true
    python_unbuffered = $true
    transcript = $TranscriptPath
    defense_command = Enable-RealtimeCondaCommand -Command $defenseCmd
    fixed_rf_attack_command = Enable-RealtimeCondaCommand -Command $fixedRfAttackCmd
    summary_command = Enable-RealtimeCondaCommand -Command $summaryCmd
}

$manifestPath = Join-Path $OutputDir "${RunName}_rfprobe_mi_sparse_v1_invocation.json"
$manifestJson = $manifest | ConvertTo-Json -Depth 8
Write-Host "Invocation manifest: $manifestPath"
Write-Host "Realtime transcript: $TranscriptPath"
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
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

    if (-not $SkipDefense) {
        Invoke-ExperimentStep -Name "Defense Stage $DefenseStage RF-probed MI sparse TAM/incoming correction" -Command $defenseCmd
    }

    if (-not $SkipFixedRfAttack) {
        Invoke-ExperimentStep -Name "Fixed RF-only attack evaluation" -Command $fixedRfAttackCmd
    }

    if (-not $SkipSummary) {
        Invoke-ExperimentStep -Name "Summarize RF-only experiment" -Command $summaryCmd
    }

    if (-not $DryRun -and (Test-Path $RunDir)) {
        Set-Content -Path (Join-Path $RunDir "rfprobe_mi_sparse_v1_invocation.json") -Value $manifestJson -Encoding UTF8
    }
} finally {
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
}

Write-Host ""
Write-Host "RF-probed MI sparse experiment run_dir: $RunDir"
Write-Host "Realtime transcript: $TranscriptPath"
