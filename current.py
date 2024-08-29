import pathlib
import random
import urllib.request
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

import pandas as pd


def segment_anything(model, paths, seed):
    masks = []
    for path in paths:
        # Load an image
        image = np.array(Image.open(path))

        # Set the image in the predictor
        model.set_image(image)
        rows, cols, _ = image.shape

        # Generate masks using a point prompt
        input_point = np.array([[int(rows * seed[0]), int(cols * seed[1])]])
        input_label = np.array([1])

        mask, _, _ = model.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=True
        )

        masks.append(mask)

    return masks


images_folder = pathlib.Path.cwd() / "assets"

data = [
    (int(file.stem), file.name, random.uniform(0.3, 1), file)
    for file in sorted(images_folder.iterdir(), key=str)
]

t = pd.DataFrame(data, columns=["id", "name", "sensitivity", "image"])

# Load the SAM model
model_type = "vit_h"
checkpoint = "sam_vit_h_4b8939.pth"

SAM_MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
model_path = checkpoint
urllib.request.urlretrieve(SAM_MODEL_URL, model_path)

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device="cpu")
predictor = SamPredictor(sam)

t = t[t["sensitivity"] >= 0.5]
t.assign(segmented=segment_anything(predictor, t["image"], [0.5, 0.6]))[
    ["id", "sensitivity", "segmented"]
].head(3)
