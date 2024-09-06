import pathlib
import random
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor


def pandas_sam_predict(model, paths, seed):
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

        mask, iou_score, _ = model.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=True
        )

        masks.append({"mask": mask[iou_score.argmax()], "iou_score": iou_score.max()})

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
df = t.assign(segmented=pandas_sam_predict(predictor, t["image"], [0.5, 0.6]))[
    ["id", "sensitivity", "segmented"]
].head(3)

# https://stackoverflow.com/a/55355928/4001592
expanded_info = pd.json_normalize(df["segmented"])
df = pd.concat([df.drop("segmented", axis=1), expanded_info], axis=1)

df = df[df["iou_score"] > 0.5]
print(df)