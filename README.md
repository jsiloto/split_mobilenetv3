# concept-drift-split-models


## Environment
To use the standardized environment on GPU 0 run:
```bash
./docker_build # this may take several minutes
./docker_run -g 0 # GPU 0
```


```bash
python train.py --checkpoint split5 --pretrained --warmup --lr 0.001 --epochs 200 --config configs/split5.yaml
```