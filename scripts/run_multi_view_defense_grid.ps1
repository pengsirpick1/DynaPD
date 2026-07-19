param(
    [ValidateSet("legacy", "x0")]
    [string]$Mode = "legacy",
    [string]$RunDir = "results\dmmpv3_attackfirst_gatedpref_seed0_20260715",
    [string]$TargetConfig = "configs\x0_target_diffusion_v1.yaml",
    [string]$OutputRoot = "",
    [double]$Budget = 0.30,
    [ValidateSet("fused", "split")]
    [string]$MultiViewMode = "fused",
    [ValidateSet("test", "val", "train", "all")]
    [string]$EvalSplit = "test",
    [int]$GenerationBatchSize = 256,
    [int]$FixedBatchSize = 256,
    [int]$Seed = 0,
    [int]$ProgressEvery = 10,
    [string]$CondaEnv = "llm",
    [ValidateSet("none", "checkpoint")]
    [string]$TeacherEvalMode = "checkpoint",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$shareTriples = @(
    "0.50,0.25,0.25",
    "0.40,0.30,0.30",
    "0.30,0.40,0.30",
    "0.30,0.30,0.40"
)

function Format-ShareLabel([string]$value) {
    return $value.Replace(".", "p")
}

foreach ($triple in $shareTriples) {
    $parts = $triple.Split(",")
    $dfShare = $parts[0]
    $awfShare = $parts[1]
    $rfShare = $parts[2]
    $label = "multi_view_${MultiViewMode}_${Mode}_df$(Format-ShareLabel $dfShare)_awf$(Format-ShareLabel $awfShare)_rf$(Format-ShareLabel $rfShare)_b$(Format-ShareLabel ([string]::Format('{0:0.00}', $Budget)))_s$Seed"

    if ($Mode -eq "legacy") {
        $script = "scripts\evaluate_legacy_pool_direct.py"
        $cmd = @(
            "run", "-n", $CondaEnv, "python", $script,
            "--run_dir", $RunDir,
            "--budget", ([string]$Budget),
            "--render_coordinate", "multi_view",
            "--multi_view_mode", $MultiViewMode,
            "--multi_view_df_share", $dfShare,
            "--multi_view_awf_share", $awfShare,
            "--multi_view_rf_share", $rfShare,
            "--tam_obfuscation_strategy", "hybrid_clustered",
            "--full_dataset",
            "--eval_split", $EvalSplit,
            "--generation_batch_size", ([string]$GenerationBatchSize),
            "--fixed_batch_size", ([string]$FixedBatchSize),
            "--progress_every", ([string]$ProgressEvery),
            "--seed", ([string]$Seed)
        )
    } else {
        $script = "scripts\evaluate_target_policy_direct_v1.py"
        $cmd = @(
            "run", "-n", $CondaEnv, "python", $script,
            "--run_dir", $RunDir,
            "--config", $TargetConfig,
            "--budget", ([string]$Budget),
            "--render_coordinate", "multi_view",
            "--multi_view_mode", $MultiViewMode,
            "--multi_view_df_share", $dfShare,
            "--multi_view_awf_share", $awfShare,
            "--multi_view_rf_share", $rfShare,
            "--tam_obfuscation_strategy", "hybrid_clustered",
            "--teacher_eval_mode", $TeacherEvalMode,
            "--full_dataset",
            "--eval_split", $EvalSplit,
            "--generation_batch_size", ([string]$GenerationBatchSize),
            "--fixed_batch_size", ([string]$FixedBatchSize),
            "--progress_every", ([string]$ProgressEvery),
            "--seed", ([string]$Seed)
        )
    }

    if ($OutputRoot.Trim()) {
        $outDir = Join-Path $OutputRoot $label
        $cmd += @("--output_dir", $outDir)
    }
    if ($Overwrite) {
        $cmd += "--overwrite"
    }

    Write-Host "[multi-view grid] running $label"
    & conda @cmd
}
