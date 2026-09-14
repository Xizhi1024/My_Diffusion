param(
    [int]$Epochs = 1000,
    [string]$ExtraOverride = ""
)

$configs = @(
    "pix2pix",
    "cyclegan",
    "reggan",
    "cpdm",
    "district_gan"
)

foreach ($name in $configs) {
    $configPath = "comparison_experiments/configs/$name.yaml"
    Write-Host "===== Training $name ====="
    if ($ExtraOverride -eq "") {
        python comparison_experiments/train_comparison.py --config $configPath --override training.num_epochs=$Epochs
    } else {
        python comparison_experiments/train_comparison.py --config $configPath --override training.num_epochs=$Epochs --override $ExtraOverride
    }
}

