param(
    [int]$Epochs = 1000,
    [int]$ImageSize = 192,
    [int]$BatchSize = 4,
    [int]$ValBatchSize = 4,
    [int]$EarlyStopPatience = 80,
    [double]$EarlyStopMinDelta = 0.0001,
    [int]$EarlyStopWarmup = 50,
    [string]$PetDir = "pet_peizhuan",
    [string]$Models = "pix2pix,cyclegan,reggan,cpdm,district_gan"
)

python comparison_experiments/run_all_png_experiments.py `
    --main-root main_data `
    --cv-root wuzhe_data `
    --models $Models `
    --epochs $Epochs `
    --image-size $ImageSize `
    --batch-size $BatchSize `
    --val-batch-size $ValBatchSize `
    --pet-dir $PetDir `
    --early-stop-patience $EarlyStopPatience `
    --early-stop-min-delta $EarlyStopMinDelta `
    --early-stop-warmup $EarlyStopWarmup
