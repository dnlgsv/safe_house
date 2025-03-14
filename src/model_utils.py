import cv2
import numpy as np
from ultralytics import YOLO

# Load model once outside function for better performance
model = YOLO(
    "yolo11n-seg.pt"
)  # TODO: create a comprohensive class for this and use config file


def predict_seg(image):
    """Perform image segmentation using YOLO model

    Args:
        image: numpy array of image

    Returns:
        tuple: (detection_status, annotated_image, description)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image to be numpy array, got {type(image)}")

    print(f"typeof image: {type(image)}")
    print(f"shape of image: {image.shape}")

    image = cv2.resize(image, (640, 640))

    results = model(image)
    objects = []

    # Access the results
    for result in results:
        # Get annotated image with bounding boxes
        annotated_img = result.plot()

        # Get class names instead of tensors
        if hasattr(result, "names") and hasattr(result, "boxes"):
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = result.names.get(class_id, f"Unknown-{class_id}")
                objects.append(class_name)

    # Create description with proper formatting and handling empty results
    if objects:
        detected = True
    else:
        detected = False

    return detected, annotated_img, objects
