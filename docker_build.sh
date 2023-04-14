#!/bin/bash
name=$(basename "$PWD")
docker build -t $USER/${name} \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg USER=$(id -un) \
  --build-arg GROUP=$(id -gn) .