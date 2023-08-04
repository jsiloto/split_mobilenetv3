import argparse
import os
import torch
import numpy as np
from dataset import get_dataset
import time

from model import get_model
from utils import load_config

parser = argparse.ArgumentParser(description='Latency Benchmarks')
parser.add_argument('--model', help='split model yaml file path')
parser.add_argument('--dataset', required=True, help='dataset name')

def exp(model, input_shape, repetitions, device):
    input_shape = (repetitions,) + input_shape
    dummy_input = torch.randn(input_shape, dtype=torch.float).to(device)
    timings = np.zeros((repetitions, 1))
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            start = time.time()
            _ = model(torch.unsqueeze(dummy_input[rep], dim=0))
            curr_time = time.time() - start
            timings[rep] = curr_time
    return timings



def benchmark_model_inference(model, input_shape, device):

    warmup_times = 200
    experiment_times = 200
    model.to(device)

    while(True):
        warmup_timings = exp(model, input_shape, warmup_times, device)
        experiment_timings = exp(model, input_shape, experiment_times, device)
        avg, std = np.average(experiment_timings), np.std(experiment_timings)
        if std < avg/5:
            break
        else:
            print("Unstable experiment -- Rerunning...")

    print(f"{round(1000*avg, 1)} ms")
    return warmup_timings, experiment_timings

def main():
    args = parser.parse_args()

    d = get_dataset(args.dataset, 1)
    input_image = d.train_dataset.__getitem__(0)[0].to("cuda:0")

    model_config = load_config(args.model)
    model = get_model(model_config, num_classes=d.num_classes)
    model.eval()
    encoder = model.encoder
    decoder = model.decoder
    encoder.codec.entropy_bottleneck.update()
    print("Full Model")
    benchmark_model_inference(model=model, input_shape=d.input_shape, device="cuda:0")

    print("Encoder Model")
    benchmark_model_inference(model=encoder, input_shape=d.input_shape, device="cuda:0")

    print("Decoder Model")
    shape = encoder(torch.unsqueeze(input_image, dim=0))['y_hat'].shape
    benchmark_model_inference(model=decoder, input_shape=shape[1:], device="cuda:0")



if __name__ == "__main__":
    main()

