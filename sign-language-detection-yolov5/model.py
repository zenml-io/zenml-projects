from pathlib import Path

import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from yolov5.utils.general import (
    check_file,
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

# Model
device = select_device("")


class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, imgs):
        source = str(imgs)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        )
        if is_url and is_file:
            source = check_file(source)  # download

        # Load model
        device = select_device("")
        (
            stride,
            names,
            pt,
        ) = (
            self.model.stride,
            self.model.names,
            self.model.pt,
        )
        imgsz = check_img_size((640, 640), s=stride)  # check image size

        half = False
        if pt:
            self.model.model.half() if half else self.model.model.float()

        # Dataloader
        dataset = LoadImages(
            source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1
        )
        bs = 1  # batch_size

        # Run inference
        self.model.warmup(
            imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz)
        )
        dt, seen = [0.0, 0.0, 0.0], 0
        imgs_res = []
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = self.model(im)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(
                prediction=pred,
                conf_thres=torch.tensor(0.25),
                iou_thres=torch.tensor(0.45),
                classes=None,
                agnostic=False,
                max_det=1000,
            )
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = (  # mypy: ignore
                    path,
                    im0s.copy(),
                    getattr(dataset, "frame", 0),
                )  # mypy: ignore
                p = Path(p)  # to Path
                s += "%gx%g " % im.shape[2:]  # print string
                torch.tensor(im0.shape)[
                    [1, 0, 1, 0]
                ]  # mypy: ignore  # mypy: ignore
                annotator = Annotator(im0, line_width=1, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(
                        im.shape[2:], det[:, :4], im0.shape
                    ).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                        # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = f"{names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        im_crop = save_one_box(
                            xyxy, im0, gain=1.1, pad=12, BGR=True, save=False
                        )
                        imgs_res.append(im_crop.tolist())
        return (s, imgs_res)


def wrapped_model(model_path):
    original_model = DetectMultiBackend(model_path, device=device, dnn=False)
    model = WrapperModel(original_model)
    return model
