# concept-drift-split-models


## Environment
To use the standardized environment on GPU 0 run:
```bash
./docker_build # this may take several minutes
./docker_run -g 0
```

```bash
python train.py --dataset stl10 --regular configs/model/regular.yaml --split configs/model/split.yaml

# Extract the embeddings as
python latency_benchmark.py --model configs/model/split.yaml --dataset stl10
```
