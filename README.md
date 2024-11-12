# Setup

```bash
conda create --name cmu_10718 python==3.11
conda activate cmu_10718
pip install -r requirements.txt

pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

To install fastFM (linux only)

```bash
pip install cython
pip install fastFM
```

# Datasets

- [CopperUnion Anime Recommendations](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

```bash
kaggle datasets download -d CooperUnion/anime-recommendations-database -p data/copperunion
tar -xf data/copperunion/anime-recommendations-database.zip -C data/copperunion

OR

unzip data/copperunion/anime-recommendations-database.zip -d data/copperunion
```

Michael is big dum dum

# Running Pipeline V2

```bash
$env:PYTHONPATH="src"; python ./src/recommender/run_pipeline.py --dataset_config configs/popularity/dataset.json --model_config configs/popularity/model.json --model POPULARITY --output_dir "models/popularity"
```
