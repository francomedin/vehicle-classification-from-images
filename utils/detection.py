import torch, cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Load here your Detection model
# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because this particular model has a good balance between accuracy and speed.
# You can check the following Colab notebook with examples on how to run
# Detectron2 models
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.
# Assign the loaded detection model to global variable DET_MODEL
# TODO

# Getting configs, weights and creating the detection model.
cfg = get_cfg()
# cfg.MODEL.DEVICE="cpu"
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)
DET_MODEL = DefaultPredictor(cfg)


def get_vehicle_coordinates(img):
    """
    This function will run an object detector over the the image, get
    the vehicle position in the picture and return it.

    Many things should be taken into account to make it work:
       1. Current model being used can detect up to 80 different objects,
          we're only looking for 'cars' or 'trucks', so you should ignore
          other detected objects.
       2. The object detector may find more than one vehicle in the picture,
          you must then, choose the one with the largest area in the image.
       3. The model can also fail and detect zero objects in the picture,
          in that case, you should return coordinates that cover the full
          image, i.e. [0, 0, width, height].
       4. Coordinates values must be integers, we're making reference to
          a position in a numpy.array, we can't use float values.

    Parameters
    ----------
    img : numpy.ndarray
       Image in RGB format.

    Returns
    -------
    box_coordinates : tuple
       Tuple having bounding box coordinates as (left, top, right, bottom).
       Also known as (x1, y1, x2, y2).
    """
    # TODO

    outputs = DET_MODEL(img)
    instances = outputs["instances"]
    # Get only class 2 and 7
    if (2 in instances.pred_classes) | (7 in instances.pred_classes):
        valid_instances = instances[
            (instances.pred_classes == 2) | (instances.pred_classes == 7)
        ]
        # Class with max area
        max_area_index = torch.argmax(valid_instances.pred_boxes.area()).item()
        box_coordinates = (
            valid_instances.pred_boxes.tensor[max_area_index].to(torch.int).tolist()
        )
    else:
        h, w = img.shape[:2]
        outputs = [0, 0, w, h]
        box_coordinates = outputs

    return box_coordinates
