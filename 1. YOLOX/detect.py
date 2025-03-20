import os
import cv2
import torch
import random
import pickle
import numpy as np
import torch.backends.cudnn as cudnn
import logging

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess

# Configure logging for detect
detect_logger = logging.getLogger('detect')

def detect(
    exp_file: str,
    ckpt_file: str,
    output_file: str,
    frame_paths: list,
    test_conf: float = 0.5,
    nmsthre: float = 0.45,
    test_size: tuple = (896, 1600),
    fuse: bool = True,
    fp16: bool = True,
    seed: int = None,
    local_rank: int = 0,
    device: str = None
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    detect_logger.info(f"Using device: {device}")

    detect_logger.info(f"Checkpoint exists: {os.path.exists(ckpt_file)}")
    detect_logger.info(f"Checkpoint size (bytes): {os.path.getsize(ckpt_file)}")
    detect_logger.info(f"Exp file exists: {os.path.exists(exp_file)}")
    detect_logger.info(f"Number of frames: {len(frame_paths)}")

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.benchmark = True

    exp = get_exp(exp_file)
    exp.test_conf = test_conf
    exp.nmsthre = nmsthre
    exp.test_size = test_size
    exp.num_classes = 1

    model = exp.get_model().cuda(local_rank).eval()
    torch.cuda.set_device(local_rank)

    ckpt = torch.load(ckpt_file, map_location=f"cuda:{local_rank}", weights_only=True)
    model.load_state_dict(ckpt["model"])
    detect_logger.info("YOLOX model loaded successfully")

    if fuse:
        model = fuse_model(model)
    if fp16:
        model = model.half()

    detections_per_frame = {}
    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            detect_logger.warning(f"Failed to load frame: {frame_path}")
            detections_per_frame[frame_idx + 1] = np.array([])
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, ratio = exp.preproc(frame_rgb, exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if fp16:
            img = img.half()

        with torch.no_grad():
            outputs = model(img)
            predictions = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)[0]

        if predictions is None or len(predictions) == 0:
            detect_logger.info(f"Frame {frame_idx + 1}: No detections")
            detections_per_frame[frame_idx + 1] = np.array([])
            continue

        boxes = predictions[:, :4].cpu().numpy() / ratio
        confidences = predictions[:, 4].cpu().numpy()
        class_ids = predictions[:, 6].cpu().numpy()
        mask = class_ids == 0
        boxes, confidences = boxes[mask], confidences[mask]

        detections = np.zeros((len(boxes), 7))
        detections[:, :4] = boxes
        detections[:, 4] = confidences
        detections[:, 5] = 0
        detections[:, 6] = 1.0
        detections_per_frame[frame_idx + 1] = detections
        detect_logger.info(f"Frame {frame_idx + 1}: Detected {len(boxes)} objects")

    with open(output_file, 'wb') as f:
        pickle.dump(detections_per_frame, f, protocol=pickle.HIGHEST_PROTOCOL)
    detect_logger.info(f"Detections saved to {output_file}")

    return detections_per_frame