#!/bin/bash

CONTAINER_NAME="gradTTS_torch"
SOURCE_CODE_MOUNT="$(pwd)":/workspace/local
LJSPEECH_MOUNT="/mnt/QNAP/staria/LJSpeech-1.1:/datasets/LJSpeech"
OUTPUTS_MOUNT="/mnt/QNAP/staria/bogdan_outputs:/outputs"

docker container run -d \
                    -it "$(id -u):$(id -g)" \
                    --name $CONTAINER_NAME \
                    -v $SOURCE_CODE_MOUNT \
                    -v $LJSPEECH_MOUNT \
                    -v $OUTPUTS_MOUNT \
                    --gpus=all \
                    nvcr.io/nvidia/pytorch:24.02-py3

docker exec -it $CONTAINER_NAME /bin/bash


# Delete a container
# docker container rm -f $CONTAINER_NAME