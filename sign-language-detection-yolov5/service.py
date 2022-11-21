import sys

import bentoml
from bentoml.io import NumpyNdarray, Text

sys.path.insert(0, "yolov5")

yolo_runner = bentoml.pytorch.get("sign_language_yolov5").to_runner()

svc = bentoml.Service(
    name="sign_language_yolov5_service",
    runners=[yolo_runner],
)


@svc.api(input=Text(), output=Text())
async def predict(img: str) -> str:
    assert isinstance(img, str)
    sign, _ = await yolo_runner.async_run(img)
    return sign


@svc.api(input=Text(), output=NumpyNdarray(dtype="float32"))
async def predict_img(img: str) -> str:
    assert isinstance(img, str)
    _, img = await yolo_runner.async_run(img)
    return img
