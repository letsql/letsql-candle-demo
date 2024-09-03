import io
import pathlib
import random
import urllib.request

import ibis.expr.datatypes as dt
import letsql as ls
import pyarrow as pa

from ibis import udf
from PIL import Image
from letsql.common.caching import SourceStorage
from letsql import _


IMAGE_FORMAT = "JPEG"


@udf.scalar.builtin
def segment_anything(path: str, img: dt.binary, s: list[float]) -> dt.Struct(
    {"mask": dt.Array[float], "iou_score": float}
):
    """Run Segment Anything in a Binary Column"""


def get_blob(path):
    image = Image.open(path)

    output = io.BytesIO()
    image.save(output, format=IMAGE_FORMAT)

    return output.getvalue()


images_folder = pathlib.Path.cwd() / "assets"

data = [
    (int(file.stem), file.name, get_blob(file))
    for file in sorted(images_folder.iterdir(), key=str)
]
ids, names, images = zip(*data)

# create and register table
table = pa.Table.from_arrays(
    [
        pa.array(ids),
        pa.array(names),
        pa.array([random.uniform(0.3, 1) for _ in range(10)], type=pa.float64()),
        pa.array(images, type=pa.binary()),
    ],
    names=["id", "name", "sensitivity", "image"],
)

con = ls.connect()
t = con.register(table, table_name="images")

# download model
SAM_MODEL_URL = "https://storage.googleapis.com/letsql-assets/models/mobile_sam-tiny-vitt.safetensors"
model_path = "mobile_sam-tiny-vitt.safetensors"
urllib.request.urlretrieve(SAM_MODEL_URL, model_path)

storage = SourceStorage(source=ls.duckdb.connect())
expr = (
    t.select(
        [
            "id",
            "sensitivity",
            segment_anything(str(model_path), t.image, [0.5, 0.6]).name("segmented"),
        ]
    )
    .filter([t.sensitivity >= 0.5])
    .limit(3)
    .cache(storage)
    .filter([_.segmented.iou_score > 0.5])
    .select([_.id, _.sensitivity, _.segmented.iou_score, _.segmented.mask])
)

result = expr.execute()
print(result)