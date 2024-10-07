# Setup

```bash
conda create --name cmu_10718 python==3.11
conda activate cmu_10718
pip install -r requirements.txt
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
