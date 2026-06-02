# TODO BEFORE THE END
- change path to repo when migrated to terranum


# Playground segmentation

## How to install
to install this pipeline, you need to have CUDA installed on your machine and follow these steps:
1) Clone the repo on local\
  Open a terminal, cd your way to the desired location and type:
    ```
    git clone --depth=1 https://github.com/destoswa/playground_segmentation
    ```

2) Create a virtual environment
    ```
    cd playground_segmentation
    python -m venv .venv
    ```
4) Install the libaries
    ```
    pip install -r requirements.txt
    ```

## Introduction
This repo contains a pipeline made to practice binary semantic segmentation. The pretrained model has been finetuned to discriminate playgrounds from background in aerial images with original resolution of 10cm/pixel.

However, those images were alternated to obtaine version with greater context and lower resolution (resolution lowered down to a factor 0.25), allowing the model to be more versatile.

This versatility allows then, in the Inference phase (production.py), to assess the samples under multiple resolutions. By doing that and regrouping the different predictions together, it allows the pipeline to ally the advantages of lower resolution and higher context while keeping the samples at a resonable size (a NVIDIA LAPTOP RTX3060 is enough to make production).

## Model
...

## Inference
...

## Training
...

## Aknowledgement
...
