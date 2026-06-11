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
3) Activate environment
   1) On Windows
        ```
        .venv\Scripts\activate
        ```
    2) On Linux
        ```
        .venv\bin\activate
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
The model used is called SegFormer. It is available on Huggingface database (see links in acknowledgements).

It's particularity is the use of Mix Transformer which allow the use of both self-attention and cnn into a computational power efficient architecture.

![segformer architecture](medias/segformer_architecture.png)

The different available pretrained model of Huggingface are listed [here](https://huggingface.co/models?other=segformer) .

## Production (Inference)
A pretrained model is already present in the repo under `model/segmenter_playground`.
In order to make predictions on new tiles, open the file `production.yaml` and precise the following:
if you already have the tiles, set their location into `data\source`.
if you want to download them on swisstopo, use the `downloader` section (`skip_auto_sownloading` -> False). the tiles will be saved in `data\source`.

A few important parameters:
- `predictions\model_dir`: Here the default one is the model comming with the repo. If you finetuned it into a new version, precise the path to the new one (not the checkpoint, but the parent of the different checkpoint. it will automatically gofind the best version)
- `predictions\batch_size`: The initial value is the one used with 6Go of VRAM. Adapat it accordingly to if you have more or less VRAM.
- `predictions\threshold_preds`: This parameter is very important. A higer value (e.g. 0.5) will lead to less but more precise predictions (less false positives but more false negatives) and a lower value (e.g. 0.25) will lead to the opposite.
- `to_keep\*`: Choose here which results you want to keep, additionaly to the final product (geopackage).

### Downloader
The downloader is a tool of the production to automatically go download tiles from the SwissImage project of Swisstopo at a resolution of 10cm / pixel.

There is 4 different modes:
- `full`: will download every tile of Switzerland (the most recent version). Carefull with this parameter! Since there is 42000 tiles covering the country and that each tile weights 65Mo, the downloading of the full country would take hours and some 2.8 To.
- `canton`: will download the most recent tiles covering the given canton. The list of accepted values is givent in the comment of the corresponding parameter in the file `config/production.yaml`.
- `area`: here you need to precise the limits of a square, in the coordinates of EPSG:2056 (only the 4 last digit since they are enough to cover the full country).
- `year`: download all tiles of a given year. This option can be interesting if we want to capture every tile of a given flight.

To enable this tool, set the parameter `downloader/skip_auto_downloading` to False.
 
## Cleaning
Once the pipeline produced predictions for every tile, those predictions need to be verified by hand. In order to do that, the methodology used in this project is the following:
1) Open the merged version of the polygons in QGIS and set it in edit mode
2) Open the plugin _*Go2NextFeature3*_ (downloadable [here](https://plugins.qgis.org/plugins/Go2NextFeature3)) with the following parameters:
   - Layer: the merged layer
   - Update state: check and select the field `Update state`
3) Go through all the prediction by setting the correct samples as "fixed" (no mater the value here. we just want to distinguish the correct from the wrong predictions)
4) In the attribute table, sort by `update_state` and select all the samples with the value "fixed".
5) Output a layer called for example _**<original_layer_name>_clean**_
6) Run the script `qgis_prepare_final_product.py` in the pyhont IDE of QGIS while giving the layer name and the output path for the final product.
   - The script will merge together the samples that are less than 20m apart and output all the centroids. 

## Training
The training of a model is done via the script `training.py` and is driven by the configuration file `config/training.yaml`.
It is possible to train a new model:
- from one of the pretrained model of NVIDIA
- from a pretrained model of this project
- from an interrupted training 

A few important parameters:
- `dataset\mode`: should be almost always on auto with the dataset location specified in `dataset\dataset_dir`. If as so, the parameters `trainset_dir` and `valset_dir` can be ignored. trainingset and validation set will be automatically generated with `val_split` fraction of the dataset to the validation set and the rest to the training set.
- `train\from_pretrain`: In most of the case, you will want to finetune the existing model. Therefore, this parameter should be set to True, with the root of the of the model specified in `pretrain_dir`.
- `train\do_save_best_preds`: If you want to see how the best version of the model performed on the validation set, you can set it to True. The results will be visible in the subfolder _logs_ of the result.

### Preprocessing
The preprocessing is a tool that help to prepare a dataset from a set of tiles of bigger size (e.g. 4100px) centered on the object of interest, to cropped version (e.g. 512px) of different scales. It is run through the script `preprocessing.yaml`, drive by teh configuration file `config\preprocessing.yaml`

As explained in the figure, 16 versions of the dataset (more precisaly, it is producing $N_{central squares} \cdot N_{scales}$ datasets) are produced. The idea is then for the user to choose, for each scale, the version with the central square which offers the best ratio of empty/occupied samples. This ratio can be seen for each configuration in the generated figure `fraction_of_occupied.png` in the resulting folder.

![preprocessing schema](medias/preprocessing.png)

### Finetuning
With the use of the model and the correction by human eye, new data should start to be produce. Moreover, some patterns of false positives might start to be visible with corrections. From these samples can be extracted a dataset to finetune the model.

In order to generate this dataset, a tutorial (the file `tuto_finetuning.md`) is available at the root of the project. It will help the user to create the tiles and then use the preprocessing script.

## Acknowledgements
This project was done by using both the [Transformers](https://huggingface.co/docs/transformers/index) library and the 
[SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) model of Huggingface.

