# Playground segmentation
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
