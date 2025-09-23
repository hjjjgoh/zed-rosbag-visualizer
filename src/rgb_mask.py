import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image

def depth_mask(pts: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Creates a boolean mask for points within a specified depth range."""
    return (pts[..., 2] > min_depth) & (pts[..., 2] < max_depth)


class ImageSegmenter:
    """
    A class to handle image segmentation for background removal using a BiRefNet model.
    """
    def __init__(self, device: str = "cuda"):
        """
        Initializes the segmenter, loads the model, and prepares the image transformations.
        
        Args:
            device (str): The device to run the model on, e.g., "cuda" or "cpu".
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"ImageSegmenter using device: {self.device}")

        torch.set_float32_matmul_precision("high")
        
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def get_mask(self, image_np: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Generates a binary background mask for a given RGB image.

        Args:
            image_np (np.ndarray): The input RGB image as a NumPy array of shape (H, W, 3).
            threshold (float): The threshold to binarize the segmentation mask.

        Returns:
            np.ndarray: A boolean mask of shape (H, W) where True indicates the foreground.
        """
        image_pil = Image.fromarray(image_np)
        original_size = image_pil.size
        
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
            
        pred_tensor = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred_tensor)
        
        mask_pil = pred_pil.resize(original_size, Image.LANCZOS)
        mask_np = np.array(mask_pil)
        
        # Normalize to 0-255 if it's not already
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)

        return mask_np > (threshold * 255)

# Create a single instance to be used throughout the application
# This prevents reloading the model every time.
try:
    segmenter = ImageSegmenter()
except Exception as e:
    print(f"Failed to initialize ImageSegmenter: {e}")
    print("Segmentation will not be available. Falling back to a dummy segmenter.")
    # Create a dummy class if initialization fails (e.g., no CUDA, no internet)
    class DummySegmenter:
        def get_mask(self, image_np: np.ndarray, threshold: float = 0.5) -> np.ndarray:
            print("WARNING: Using dummy segmenter. Returning a mask of all True.")
            return np.ones((image_np.shape[0], image_np.shape[1]), dtype=bool)
    segmenter = DummySegmenter()
    