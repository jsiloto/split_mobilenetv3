# concept-drift-split-models


## Environment
To use the standardized environment on GPU 0 run:
```bash
./docker_build # this may take several minutes
./docker_run -g 0
```

Download example checkpoints
```bash
wget --no-check-certificate 'https://drive.google.com/u/0/uc?export=download&confirm=1wnBBQYG21b_rvGlmDi9kboTxcDlbRV2K' -O checkpoints.zip
https://drive.google.com/file/d/1wnBBQYG21b_rvGlmDi9kboTxcDlbRV2K/view?usp=sharing

```

Run Training and Latency Benchmarks
```bash
python train.py --dataset oxford_pets --model configs/model/split.yaml
python latency_benchmark.py --model configs/model/split.yaml --dataset oxford_pets
```
