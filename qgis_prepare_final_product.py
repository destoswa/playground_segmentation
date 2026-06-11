import processing
import os
from qgis.core import QgsProject, QgsVectorFileWriter, QgsCoordinateTransformContext

# --------------------------------
# SETTINGS
# --------------------------------
layer_name = "clean_version_0.5"   # change this to your layer name
output_path = r"D:\GitHubProjects\Terranum_repo\playground_segmentation\data\test_20_tuiles\th=0.5\final_product.gpkg"  # change this
buffer_dist = 10

# --------------------------------
# Get layer
# --------------------------------
layer = QgsProject.instance().mapLayersByName(layer_name)[0]

# --------------------------------
# Buffer +10m with dissolve (disjoint separate)
# --------------------------------
buf_pos = processing.run("native:buffer", {
    'INPUT': layer,
    'DISTANCE': buffer_dist,
    'SEGMENTS': 5,
    'END_CAP_STYLE': 0,
    'JOIN_STYLE': 0,
    'MITER_LIMIT': 2,
    'DISSOLVE': True,
    'SEPARATE_DISJOINT': True,
    'OUTPUT': 'memory:'
})['OUTPUT']

# --------------------------------
# Buffer -10m
# --------------------------------
buf_neg = processing.run("native:buffer", {
    'INPUT': buf_pos,
    'DISTANCE': -buffer_dist,
    'SEGMENTS': 5,
    'END_CAP_STYLE': 0,
    'JOIN_STYLE': 0,
    'MITER_LIMIT': 2,
    'DISSOLVE': False,
    'OUTPUT': 'memory:'
})['OUTPUT']

# --------------------------------
# Centroids
# --------------------------------
centroids = processing.run("native:centroids", {
    'INPUT': buf_neg,
    'ALL_PARTS': False,
    'OUTPUT': 'memory:'
})['OUTPUT']

# --------------------------------
# Remove fid field
# --------------------------------
fid_index = centroids.fields().indexFromName('fid')
if fid_index != -1:
    centroids = processing.run("qgis:deletecolumn", {
        'INPUT': centroids,
        'COLUMN': ['fid'],
        'OUTPUT': 'memory:'
    })['OUTPUT']

# --------------------------------
# Add unique id field (not named fid)
# --------------------------------
centroids = processing.run("native:addautoincrementalfield", {
    'INPUT': centroids,
    'FIELD_NAME': 'uid',
    'START': 1,
    'OUTPUT': 'memory:'
})['OUTPUT']

# --------------------------------
# Export to GPKG
# --------------------------------
QgsVectorFileWriter.writeAsVectorFormatV2(
    centroids,
    output_path,
    QgsCoordinateTransformContext(),
    QgsVectorFileWriter.SaveVectorOptions()
)

print(f"Done! Centroids saved to {output_path}")