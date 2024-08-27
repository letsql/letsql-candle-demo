import letsql as ls
import pyarrow as pa
import io
import pathlib
from PIL import Image
import urllib.request

IMAGE_FORMAT = "JPEG"


def get_blob(path):
    image = Image.open(path)

    output = io.BytesIO()
    image.save(output, format=IMAGE_FORMAT)

    return output.getvalue()


images_folder = pathlib.Path.cwd() / "assets"

data = [(int(file.stem), file.name, get_blob(file)) for file in sorted(images_folder.iterdir(), key=str)]
ids, names, images = zip(*data)
print(names)

table = pa.Table.from_arrays(
    [
        pa.array(ids),
        pa.array(names),
        pa.array(images, type=pa.binary()),
    ],
    names=["id", "name", "data"],
)

# create table
con = ls.connect()
images = con.register(table, table_name="images")

# download model
SAM_MODEL_URL = "https://storage.googleapis.com/letsql-assets/models/mobile_sam-tiny-vitt.safetensors"
model_path = "mobile_sam-tiny-vitt.safetensors"
urllib.request.urlretrieve(SAM_MODEL_URL, model_path)


expr = images.data.segment_anything(str(model_path), [0.5, 0.6]).name("segmented")
result = expr.execute()


for i, data in enumerate(result.to_list()):
    image = Image.open(io.BytesIO(data))
    image.save(f"output_{i}.jpeg")
