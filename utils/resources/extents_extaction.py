from qgis.core import QgsMapSettings, QgsMapRendererParallelJob, QgsRectangle
from qgis.PyQt.QtCore import QSize
import os


# === PARAMETERS ===
output_source = "path/to/output_dataset"    # where the resulting dataset will be created
tiles_rectangle_layer_name = "tiles_410m"   # the name of the layer with the rectangles bboxing the tiles
what_to_process = 'images'                  # choose between 'images' and 'masks' depending on the step you're at
do_skip_existing = True                     # usefull if QGIS failed and you want to rerun a partially failed process
# ==================


assert what_to_process in ['images', 'masks']

src_extents = os.path.join(output_source, what_to_process)
os.makedirs(src_extents, exist_ok=True)

layer = QgsProject.instance().mapLayersByName(tiles_rectangle_layer_name)[0]
canvas_layers = iface.mapCanvas().layers()  # get ONCE

for counter, feat in enumerate(layer.getFeatures()):
    out_path = os.path.join(src_extents, f"image_{feat['id'].replace('/', '_')}.tif")
    if os.path.exists(out_path) and do_skip_existing:
        continue

    # ✅ Direct bbox from feature
    rect: QgsRectangle = feat.geometry().boundingBox()

    width_m = rect.width()
    height_m = rect.height()

    img_size = QSize(int(width_m * 10), int(height_m * 10))

    settings = QgsMapSettings()
    settings.setOutputSize(img_size)
    settings.setExtent(rect)
    settings.setLayers(canvas_layers)
    settings.setDestinationCrs(layer.crs())

    job = QgsMapRendererParallelJob(settings)
    job.start()
    job.waitForFinished()

    image = job.renderedImage()
    image.save(out_path)

    print("Saved:", out_path)
