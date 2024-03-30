#!/bin/bash

CONTAINER_NAME="gradTTS_torch"
SOURCE_CODE_MOUNT="$(pwd)":/workspace/local
LJSPEECH_MOUNT="/datasets/LJSpeech:/datasets/LJSpeech"
OUTPUTS_MOUNT="/home/astanea/outputs/GradTTS:/outputs"

docker container run -d \
                    -it \
                    --name $CONTAINER_NAME \
                    -v $SOURCE_CODE_MOUNT \
                    -v $LJSPEECH_MOUNT \
                    -v $OUTPUTS_MOUNT \
                    --gpus=all \
                    nvcr.io/nvidia/pytorch:24.02-py3

docker exec -it $CONTAINER_NAME /bin/bash

# Delete a container
# docker container rm -f $CONTAINER_NAME