import torch
import cv2


def load_midas_model(model_type="MiDaS_small"):
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    return midas, transform


def estimate_depth(model, transform, image):
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(input_image)
    with torch.no_grad():
        depth = model(input_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=input_image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    return depth.cpu().numpy()