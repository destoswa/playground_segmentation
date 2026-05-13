import os
import torch
import rasterio
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from torchvision.transforms.functional import to_pil_image
import json


# def load_latest_checkpoint(model_dir):
#     """
#     Returns the path to the latest checkpoint folder inside model_dir.
#     If none found, return model_dir (trained_model directory).
#     """
#     if not os.path.isdir(model_dir):
#         raise ValueError(f"Model directory not found: {model_dir}")

#     ckpts = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
#     if not ckpts:
#         print("[INFO] No checkpoints found. Using main model directory.")
#         return model_dir

#     # Sort by step number
#     ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
#     last_ckpt = ckpts_sorted[-1]

#     print(f"[INFO] Using checkpoint: {last_ckpt}")
#     return os.path.join(model_dir, last_ckpt)

def load_latest_checkpoint(model_dir, verbose=False):
    """
    Retrieve the latest checkpoint directory inside a model directory.
    Parameters:
        model_dir (str) - directory containing model checkpoints.
        verbose (bool) - whether to print information about the selected checkpoint.
    Returns:
        str - path to the latest checkpoint folder or the base model directory if none found.
    """
    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    ckpts = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    if not ckpts and verbose:
        print("[INFO] No checkpoints found. Using main model directory.")
        return model_dir

    # Sort by step number
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
    last_ckpt = ckpts_sorted[-1]
    with open(os.path.join(model_dir, 'last_checkpoint/trainer_state.json'), "r") as jsonfile: 
        infos = json.load(jsonfile)
    best_model = infos['best_model_checkpoint']
    if verbose:
        print(f"[INFO] Using checkpoint: {best_model}")
    return best_model
    return os.path.join(model_dir, last_ckpt)


def predict_image(model, processor, image, device="cuda"):
    """
    Runs inference on a single image and returns:
    - predicted_mask (H, W) with class indices
    """
    # test if image is a path or already an Image.Image object
    if not isinstance(image, Image.Image):
        image = rasterio.open(image).read()[:3, ...]
        image = torch.tensor(image, device=device).unsqueeze(0)
    width, height = image.shape[-2::]

    # Preprocess
    # inputs = processor(image, return_tensors="pt").to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    image = (image - mean) / std
    with torch.no_grad():
        # outputs = model(**inputs)
        outputs = model(image)
        logits = outputs.logits  # (1, num_classes, h, w)

    # Resize logits to original size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )

    pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    return pred_mask, upsampled_logits


def run_inference(conf):
    """
    Runs inference on all images in a directory.
    """
    # ----------------------------
    # Load parameters
    # ----------------------------
    MODEL_DIR = conf.model_dir
    DATA_DIR = conf.data_dir
    SAVE_MASK_AS_IMG = conf.save_mask_as_img
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ----------------------------
    # Load best checkpoint
    # ----------------------------
    ckpt_path = load_latest_checkpoint(MODEL_DIR)

    # processor = AutoImageProcessor.from_pretrained(ckpt_path)
    processor = None
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt_path)
    model.to(DEVICE)
    model.eval()

    OUTPUT_DIR = os.path.join(DATA_DIR, "predictions")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_DIR_AS_IMG = os.path.join(OUTPUT_DIR, "images") if SAVE_MASK_AS_IMG else None
    os.makedirs(OUTPUT_DIR_AS_IMG, exist_ok=True)

    # ----------------------------
    # Loop over images
    # ----------------------------
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    image_list = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(exts)]

    if not image_list:
        print("No images found in:", DATA_DIR)
        return

    print(f"[INFO] Running inference on {len(image_list)} images")

    for _, img_name in tqdm(enumerate(image_list), total=len(image_list), desc="Predicting"):
        input_path = os.path.join(DATA_DIR, img_name)
        output_path = os.path.join(OUTPUT_DIR, '.'.join(img_name.split('.')[:-1]) + ".tif")

        mask, preds = predict_image(model, processor, input_path, device=DEVICE)
        pil_mask = Image.fromarray(mask.astype(np.uint8))
        pil_mask.save(output_path)

        if SAVE_MASK_AS_IMG:
            rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
            rgb_mask[mask == 1] = 255
            pil_rgb_mask = Image.fromarray(rgb_mask.astype(np.uint8))
            pil_rgb_mask.save(os.path.join(OUTPUT_DIR_AS_IMG, img_name))


if __name__ == "__main__":
    conf = OmegaConf.load('config/inference.yaml')

    run_inference(conf)
