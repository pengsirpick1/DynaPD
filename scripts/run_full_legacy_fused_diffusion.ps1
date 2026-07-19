param(
    [string]$RunName = "",
    [string]$DataRoot = "D:\learning\TOR\datasets\CW",
    [string]$OutputDir = "results",
    [int]$Seed = 0,
    [string]$CondaEnv = "llm",
    [double]$Budget = 0.30,
    [int]$DiffusionTrainSteps = 30000,
    [int]$EncoderEpochs = 10,
    [int]$SurrogateEpochs = 10,
    [int]$AttackEpochs = 10,
    [int]$AdaptiveEpochs = 10,
    [int]$BatchSize = 128,
    [int]$AttackBatchSize = 256,
    [int]$DeploymentRepeats = 3,
    [ValidateSet("same_user", "multi_source", "full_catalogue", "cross_user", "profile_known")]
    [string[]]$AdaptiveProtocols = @("same_user", "full_catalogue"),
    [int]$SourceUserCount = 4,
    [switch]$SkipDefense,
    [switch]$SkipFixedAttack,
    [switch]$SkipAdaptiveAttack,
    [switch]$ForceRetrainAttack,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not $RunName.Trim()) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunName = "dmmpv3_legacydirect_fused_fullcw_seed${Seed}_${stamp}"
}

$RunDir = Join-Path $OutputDir $RunName
$BudgetText = [string]::Format("{0:0.00}", $Budget)

function Invoke-ExperimentStep {
    param(
        [string]$Name,
        [string[]]$Command
    )
    $RealtimeCommand = New-Object System.Collections.Generic.List[string]
    $RealtimeCommand.AddRange([string[]]$Command)
    if ($RealtimeCommand.Count -gt 0 -and $RealtimeCommand[0] -eq "run" -and -not $RealtimeCommand.Contains("--no-capture-output")) {
        $RealtimeCommand.Insert(1, "--no-capture-output")
    }
    $PythonIndex = $RealtimeCommand.IndexOf("python")
    if ($PythonIndex -ge 0 -and ($PythonIndex -eq ($RealtimeCommand.Count - 1) -or $RealtimeCommand[$PythonIndex + 1] -ne "-u")) {
        $RealtimeCommand.Insert($PythonIndex + 1, "-u")
    }
    Write-Host ""
    Write-Host "========== $Name =========="
    Write-Host ("conda " + (($RealtimeCommand.ToArray()) -join " "))
    if (-not $DryRun) {
        $env:PYTHONUNBUFFERED = "1"
        $env:PYTHONIOENCODING = "utf-8"
        & conda @($RealtimeCommand.ToArray())
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment step failed with exit code $LASTEXITCODE`: $Name"
        }
    }
}

$defenseCmd = @(
    "run", "-n", $CondaEnv, "python", "scripts\train_defense.py",
    "--version", "v3",
    "--stage", "all",
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
    "--pareto_samples", "0",
    "--refine_keep_ratios", "1.0",
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
    "--render_coordinate", "multi_view",
    "--multi_view_mode", "fused",
    "--multi_view_df_share", "0.50",
    "--multi_view_awf_share", "0.25",
    "--multi_view_rf_share", "0.25",
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
    "--guidance_attackers", "both",
    "--surrogate_df_weight", "0.50",
    "--surrogate_rf_weight", "0.50",
    "--surrogate_robust_weight", "0.35",
    "--encoder_train_samples", "1000000",
    "--encoder_epochs", ([string]$EncoderEpochs),
    "--diffusion_train_steps", ([string]$DiffusionTrainSteps),
    "--batch_size", ([string]$BatchSize),
    "--probe_samples", "1000000",
    "--probe_exact_samples", "256",
    "--view_profile_samples", "0",
    "--candidate_mode", "executable",
    "--candidate_topk", "80",
    "--candidate_soft_topk",
    "--candidate_temperature", "0.15",
    "--policy_generator", "diffusion",
    "--sampling_steps", "20",
    "--guidance_weight", "0.10",
    "--guidance_last_steps", "4",
    "--guidance_train_steps", ([string]$DiffusionTrainSteps),
    "--defense_hard_weight", "1.0",
    "--defense_soft_objective_scale", "0.05",
    "--defense_soft_utility_weight", "0.05",
    "--defense_risk_tolerance", "0.0",
    "--prior_leak_weight", "1.50",
    "--prior_preference_weight", "0.15",
    "--prior_noise_std", "0.0",
    "--preference_weight", "0.01",
    "--preference_attack_gate",
    "--preference_attack_gate_margin", "0.02",
    "--constraint_weight", "0.02",
    "--profile_weight", "0.0",
    "--refine_method", "continuous",
    "--refine_steps", "6",
    "--direction_target", "none",
    "--direction_correction_strength", "0.0",
    "--deployment_repeats", ([string]$DeploymentRepeats),
    "--stage3_repeats", "1",
    "--stage3_fixed_probe_samples", "0",
    "--no-stage3_require_quality_gate",
    "--preserve_variable_length_traces",
    "--progress",
    "--log_every", "500"
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
    "--adaptive_epochs", ([string]$AdaptiveEpochs),
    "--adaptive_patience", "3",
    "--adaptive_init", "checkpoint",
    "--df_batch_size", ([string]$AttackBatchSize),
    "--df_architecture", "project",
    "--rf_tam_num_slots", "1800",
    "--max_load_time", "80.0",
    "--no-attack_require_quality_gate",
    "--progress",
    "--log_every", "500"
)

$fixedAttackCmd = @(
    "run", "-n", $CondaEnv, "python", "scripts\run_attack_eval.py",
    "--attackers", "fixed_df,fixed_rf",
    "--adaptive_protocol", "fixed"
) + $attackCommon
if ($ForceRetrainAttack) {
    $fixedAttackCmd += "--force_retrain"
}

$adaptiveCommands = @()
foreach ($protocol in $AdaptiveProtocols) {
    $cmd = @(
        "run", "-n", $CondaEnv, "python", "scripts\run_attack_eval.py",
        "--attackers", "mixed_df,mixed_rf",
        "--adaptive_protocol", $protocol,
        "--source_user_count", ([string]$SourceUserCount)
    ) + $attackCommon
    if ($ForceRetrainAttack) {
        $cmd += "--force_retrain"
    }
    $adaptiveCommands += ,@($protocol, $cmd)
}

$manifest = [ordered]@{
    run_name = $RunName
    run_dir = $RunDir
    budget = $Budget
    full_cw = $true
    defense_command = $defenseCmd
    fixed_attack_command = $fixedAttackCmd
    adaptive_protocols = $AdaptiveProtocols
    adaptive_attack_commands = @($adaptiveCommands | ForEach-Object { [ordered]@{ protocol = $_[0]; command = $_[1] } })
}

$manifestPath = Join-Path $OutputDir "${RunName}_full_legacy_fused_invocation.json"
$manifestJson = $manifest | ConvertTo-Json -Depth 8
Write-Host "Invocation manifest: $manifestPath"
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Set-Content -Path $manifestPath -Value $manifestJson -Encoding UTF8
}

if (-not $SkipDefense) {
    Invoke-ExperimentStep -Name "Defense Stage 1+2+3 full CW legacy-direct modes + fused renderer" -Command $defenseCmd
}

if (-not $SkipFixedAttack) {
    Invoke-ExperimentStep -Name "Full fixed DF/RF attack evaluation" -Command $fixedAttackCmd
}

if (-not $SkipAdaptiveAttack) {
    foreach ($entry in $adaptiveCommands) {
        Invoke-ExperimentStep -Name "Full adaptive mixed DF/RF attack evaluation: $($entry[0])" -Command $entry[1]
    }
}

$protocolCsv = "fixed"
if (-not $SkipAdaptiveAttack -and $AdaptiveProtocols.Count -gt 0) {
    $protocolCsv = "fixed," + ($AdaptiveProtocols -join ",")
}
$summaryCmd = @(
    "run", "-n", $CondaEnv, "python", "scripts\summarize_full_legacy_fused_diffusion.py",
    "--run_dir", $RunDir,
    "--protocols", $protocolCsv
)
Invoke-ExperimentStep -Name "Summarize full experiment" -Command $summaryCmd

if (-not $DryRun -and (Test-Path $RunDir)) {
    Set-Content -Path (Join-Path $RunDir "full_legacy_fused_invocation.json") -Value $manifestJson -Encoding UTF8
}

Write-Host ""
Write-Host "Full legacy-fused diffusion experiment run_dir: $RunDir"
