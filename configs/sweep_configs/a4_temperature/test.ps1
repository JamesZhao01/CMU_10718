$target = "a4_temperature";
$run1 = "base_512_temperature";
$env:PYTHONPATH="src"; 

# Run from workspace directory (CMU_10718)

python "./src/recommender/sweep_pipeline.py" `
    --dataset_config "configs/datasets/masked_is_negative.json" `
    --model_config "configs/sweep_configs/${target}/${run1}.json" `
    --sweep_config "configs/sweep_configs/${target}/${run1}_sweep.json" `
    --model "TOWER" `
    --output_dir "models/sweep/${target}/${run1}" `
    --force_retrain `
    --num_chunks 1 `
    --chunk_idx 0 `
    --do_test 