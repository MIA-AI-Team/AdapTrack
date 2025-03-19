import os
import cv2
import torch
import random
import pickle
import numpy as np
import torch.backends.cudnn as cudnn

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.models import YOLOX as YOLOXModel

def detect(
    exp_file: str,
    ckpt_file: str,
    output_file: str,
    frame_paths: list,
    test_conf: float = 0.5,
    nmsthre: float = 0.45,
    test_size: tuple = (1088, 1088),
    fuse: bool = True,
    fp16: bool = True,
    seed: int = None,
    local_rank: int = 0,
    device: str = None
):
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Verify paths
    print("Checkpoint exists:", os.path.exists(ckpt_file))
    print("Checkpoint size (bytes):", os.path.getsize(ckpt_file))
    print("Exp file exists:", os.path.exists(exp_file))
    print(f"Number of frames: {len(frame_paths)}")

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    cudnn.benchmark = True

    # Load experiment
    try:
        exp = get_exp(exp_file)
    except Exception as e:
        print(f"Failed to load exp file: {e}")
        print("Falling back to default yolox_x config")
        exp = get_exp(exp_name="yolox_x")

    exp.num_classes = 1
    exp.test_conf = test_conf
    exp.nmsthre = nmsthre
    exp.test_size = test_size

    # Load model
    model = exp.get_model()
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model.eval()

    # Load checkpoint
    try:
        ckpt = torch.load(ckpt_file, map_location=f"cuda:{local_rank}")
        state_dict = ckpt["model"]
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        raise

    # Adjust state dict keys (remove extra 'backbone.')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.backbone.backbone.'):
            new_key = k.replace('backbone.backbone.backbone.', 'backbone.backbone.')
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    print("YOLOX model loaded successfully")

    if fuse:
        model = fuse_model(model)
    if fp16:
        model = model.half()

    # Detection loop
    detections_per_frame = {}
    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to load frame: {frame_path}")
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
            detections_per_frame[frame_idx + 1] = np.array([])
            continue

        boxes = predictions[:, :4].cpu().numpy() / ratio
        confidences = predictions[:, 4].cpu().numpy()
        class_ids = predictions[:, 6].cpu().numpy()
        mask = class_ids == 0
        boxes = boxes[mask]
        confidences = confidences[mask]

        detections = np.zeros((len(boxes), 7))
        detections[:, :4] = boxes
        detections[:, 4] = confidences
        detections[:, 5] = 0
        detections[:, 6] = 1.0

        detections_per_frame[frame_idx + 1] = detections

    with open(output_file, 'wb') as f:
        pickle.dump(detections_per_frame, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Detections saved to {output_file}")

    print("Sample detections from first frame with detections:")
    found = False
    for frame_id, dets in detections_per_frame.items():
        if len(dets) > 0:
            print(f"Frame {frame_id}: {dets[:5]}")
            found = True
            break
    if not found:
        print("No detections found in any frame.")

    return detections_per_frame