# Georgia Tech OMSCS CS7643: Final Project
## Super Mario Bros - Deep Reinforcement Learning with Proximal Policy Optimization (PPO)

https://user-images.githubusercontent.com/43353147/236643104-99cf7419-748d-47b5-9b55-d63c89b6add5.mp4

### Setup
In a python 3.7+ environment, run:
```
conda create --name mario python=3.9
conda activate mario
pip install torch==1.10.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
pip install -r requirements.txt
```

### Yaml Configuration
Inside the file `configs/ppo.yaml` there are few keys that change the way the application runs.
First you must train a model so only `train` is set to true. Then once you have a model saved,
you can set `train` to `False`, `test` to `True`, then add a key `load_model` with the path to
your model found in `checkpoints/PPO/`. If you would like to generate plots you must add keys
`test_csv_path` and `train_csv_path` pointing to the csv you generate from running your training
and your test runs.

### Run
```
conda activate mario
python main.py
```

### Check nvidia memory usage
```
nvidia-smi -l --query-gpu=timestamp,temperature.gpu,memory.used,memory.total --format=csv
```

### Credits
Adopted the GAE calculation from https://github.com/uvipen/Super-mario-bros-PPO-pytorch
