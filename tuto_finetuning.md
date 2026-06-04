# Tutorial - Finetuning with new samples

## 1) Setting up
In order to produce the training set you need: 
<ol type="a">
  <li>QGIS with a map covering all the AoI</li>
  <li>A layer with the centroids of the AoI
    <ul>
      <li>If just a list of True Negatives (samples that don't contain playground), create a layer of type <b><i>Point</i></b>.</li>
      <li>If from a list of polygons, you can use the tool <b><i>Vector geometry -> Centroids</i></b></li>
    </ul>
  </li>
  <li>If finetuning on True Positives (i.e. playgrounds that were not found by the model), a layer with the polygons of each playground in full white.</li>
  <li>A layer with one black polygon covering all the AoI</li>
</ol>

## 2) Extract Images
- Create a layer, named **_bboxes_**, with Rectangles of fixed size, centered on the centroids.
  - e.g. 410m $\cdot$ 10px/m -> 4100px tiles
  - This is done by using the tool **_Vector geometry -> Rectangles, ovals, diamonds_**
- Show only the map (a), disable everything else
- Run the script `extents_extraction.py` in QGIS's python IDE
  - Adapt parameters at the beginning of the script
  - Can take a lot of time

## 3) Extract masks
- Show only black background (d) with either nothing else (if producing True negatives) or the polygons (c) on top of it.
- **IMPORTANT**: Do NOT just cover the map, **Disable it!** The process will be 100x faster.
- Run the script `extents_extraction.py` (think to change the `what_to_process` parameter)

## 4) Preprocessing
- Run the preprocessing.py script on the dataset.
  - in the config file, give the path to the root of the folders *iamges* and *masks*.
  - The preprocessing is going to create:
    -  num_scales * num_base_squares versions of the dataset
    -  an empty architecture
    -  a plot of the fraction fosamples containing playground.
- Choose one dataset per scale and place it in the empty architecture (both images & masks)

## 5) Training
- You can then finetune the model by training it from pretrained with this new dataset.

## 6) WARNINGS
- The training of a model with a non-exhaustive dataset can be tricky.
- Do not train for too many epochs! 2-5 is usually way enough.
- You might need to lower the learning rate, such that the model does not change too much.
- All this is especially true if you are training with empty samples. The model will quickly learn to always predict zero everywhere. In these cases, you might need to place also a few samples with true positives (and their corresponding mask ofc).