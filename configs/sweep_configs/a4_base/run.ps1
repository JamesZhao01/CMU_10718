$target = "a4_base";
$run1 = "base_256";
$run2 = "base_512";
$run3 = "base_512_200"; 
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
    --do_train `
    --do_test &&

python "./src/recommender/sweep_pipeline.py" `
    --dataset_config "configs/datasets/masked_is_negative.json" `
    --model_config "configs/sweep_configs/${target}/${run2}.json" `
    --sweep_config "configs/sweep_configs/${target}/${run2}_sweep.json" `
    --model "TOWER" `
    --output_dir "models/sweep/${target}/${run2}" `
    --force_retrain `
    --num_chunks 1 `
    --chunk_idx 0 `
    --do_train `
    --do_test &&

python "./src/recommender/sweep_pipeline.py" `
    --dataset_config "configs/datasets/masked_is_negative.json" `
    --model_config "configs/sweep_configs/${target}/${run3}.json" `
    --sweep_config "configs/sweep_configs/${target}/${run3}_sweep.json" `
    --model "TOWER" `
    --output_dir "models/sweep/${target}/${run3}"`
    --force_retrain `
    --num_chunks 1 `
    --chunk_idx 0 `
    --do_train `
    --do_test