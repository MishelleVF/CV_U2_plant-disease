import cv2
import numpy as np
from Operations import *

class PreprocessingPipelineConfig:
    """Configuration for the preprocessing stage.
    
    Controls: smoothing/blurring operations.
    """
    
    def __init__(
        self,
        smooth: bool = False,
        smooth_kernel_size: tuple = (5, 5),
        smooth_sigma: float = 0.0,
    ):
        """Initialize preprocessing config.
        
        Args:
            smooth: Enable Gaussian blur preprocessing
            smooth_kernel_size: Kernel size for Gaussian blur (must be odd)
            smooth_sigma: Standard deviation for Gaussian blur. 0 = auto-calculated
        """
        self.smooth = smooth
        self.smooth_kernel_size = smooth_kernel_size
        self.smooth_sigma = smooth_sigma
    
    def __repr__(self) -> str:
        return (
            f"PreprocessingPipelineConfig("
            f"smooth={self.smooth}, "
            f"smooth_kernel_size={self.smooth_kernel_size}, "
            f"smooth_sigma={self.smooth_sigma})"
        )

class MaskingPipelineConfig:
    """Configuration for the masking stage.
    
    Controls: multi-channel binarization and morphological operations.
    
    Channel specs:
    - String format: "R", "S", "AN" where letter is channel, optional N means inverted
    - Tuple format: ("R", 100) where second value is explicit threshold (no Otsu)
    """
    
    def __init__(
        self,
        binarize_rgb_channels: list = None,
        binarize_hsv_channels: list = None,
        binarize_lab_channels: list = None,
        morph_operations: list = None,
    ):
        """Initialize masking config.
        
        Args:
            binarize_rgb_channels: RGB channels to binarize (default: [])
            binarize_hsv_channels: HSV channels to binarize (default: ["S"])
            binarize_lab_channels: LAB channels to binarize (default: ["AN"])
            morph_operations: List of (name, kernel_size, cv2.MORPH_*) tuples
        """
        self.binarize_rgb_channels = binarize_rgb_channels if binarize_rgb_channels is not None else []
        self.binarize_hsv_channels = binarize_hsv_channels if binarize_hsv_channels is not None else ["S"]
        self.binarize_lab_channels = binarize_lab_channels if binarize_lab_channels is not None else ["AN"]
        self.morph_operations = morph_operations if morph_operations is not None else []
    
    def add_morph_operation(self, kernel_size: int, morph_type: int, name: str = "morph") -> None:
        """Add a morphological operation.
        
        Args:
            kernel_size: Size of the structuring element (creates kernel_size x kernel_size)
            morph_type: OpenCV morphological operation (cv2.MORPH_ERODE, cv2.MORPH_DILATE, etc.)
            name: Optional name for logging/debugging
        """
        self.morph_operations.append((name, kernel_size, morph_type))
    
    def __repr__(self) -> str:
        return (
            f"MaskingPipelineConfig("
            f"rgb={self.binarize_rgb_channels}, "
            f"hsv={self.binarize_hsv_channels}, "
            f"lab={self.binarize_lab_channels}, "
            f"morph_ops={len(self.morph_operations)})"
        )

class EdgeDetectionPipelineConfig:
    """Configuration for the edge detection stage."""
    
    def __init__(
        self,
        method: str = "sobel",
        canny_thresholds: tuple = (100, 200),
    ):
        self.method = method  # "sobel" or "canny"
        self.canny_thresholds = canny_thresholds  # only used for "canny"
    
    def __repr__(self) -> str:
        return (
            f"EdgeDetectionPipelineConfig("
            f"method={self.method!r}, "
            f"canny_thresholds={self.canny_thresholds})"
        )

class CornerDetectionPipelineConfig:
    """Configuration for the corner detection stage."""
    
    def __init__(
        self,
        method: str = "harris",
        harris_block_size: int = 2,
        harris_ksize: int = 3,
        harris_k: float = 0.04,
        shi_tomasi_max_corners: int = 100,
        shi_tomasi_quality_level: float = 0.01,
        shi_tomasi_min_distance: int = 10,
        shi_tomasi_use_harris_detector: bool = False,
    ):
        self.method = method  # "harris" or "shi-tomasi"
        self.harris_block_size = harris_block_size
        self.harris_ksize = harris_ksize
        self.harris_k = harris_k
        self.shi_tomasi_max_corners = shi_tomasi_max_corners
        self.shi_tomasi_quality_level = shi_tomasi_quality_level
        self.shi_tomasi_min_distance = shi_tomasi_min_distance
        self.shi_tomasi_use_harris_detector = shi_tomasi_use_harris_detector
    
    def __repr__(self) -> str:
        return (
            f"CornerDetectionPipelineConfig("
            f"method={self.method!r}, "
            f"harris_block_size={self.harris_block_size}, "
            f"harris_ksize={self.harris_ksize}, "
            f"harris_k={self.harris_k}, "
            f"shi_tomasi_max_corners={self.shi_tomasi_max_corners}, "
            f"shi_tomasi_quality_level={self.shi_tomasi_quality_level}, "
            f"shi_tomasi_min_distance={self.shi_tomasi_min_distance}, "
            f"shi_tomasi_use_harris_detector={self.shi_tomasi_use_harris_detector})"
        )
        

class FeatureExtractionPipelineConfig:
    """Configuration for feature extraction stage."""
    def __init__(self, methods: list = None):
        # supported: "custom", "hog", "LBP", "BoW"
        self.methods = methods if methods is not None else ["custom"]

    def __repr__(self) -> str:
        return f"FeatureExtractionPipelineConfig(methods={self.methods})"



def RunPreprocessingOnOne(img, config):
    # Apply the preprocessing operations in the order they are defined in the config
    if config.smooth:
        img = cv2.GaussianBlur(img, config.smooth_kernel_size, config.smooth_sigma)
    return img

def _binarize_channels(img, channel_specs, channel_picker):
    """Helper: binarize multiple channel specs from a single colorspace.
    
    Args:
        img: Input image in target colorspace
        channel_specs: List of channel specs (strings like "S" or tuples like ("R", 100))
        channel_picker: Function to extract channel (pickRGBChannel, pickHSVChannel, etc.)
    
    Returns:
        Combined binary mask from all channels (ORed together)
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for channel_spec in channel_specs:
        ch = None
        n = False
        threshold = None
        
        # Parse channel spec
        if isinstance(channel_spec, tuple):
            ch_str, threshold = channel_spec
            ch = ch_str[0]
            n = len(ch_str) > 1 and ch_str[1] == 'N'
        else:
            ch = channel_spec[0]
            n = len(channel_spec) > 1 and channel_spec[1] == 'N'
        
        # Extract channel and calculate threshold if needed
        channel_img = channel_picker(img, ch)
        if threshold is None:
            threshold = otsu_thresh(channel_img)
        
        # Apply threshold and inversion
        channel_mask = thresh(channel_img, threshold)
        if n:
            channel_mask = 255 - channel_mask
        
        # Combine via OR
        mask = cv2.bitwise_or(mask, channel_mask)
    
    return mask


def RunMaskingOnOne(img, config):
    """Apply masking pipeline to a single image.
    
    Binarizes multi-channel colorspaces (RGB, HSV, LAB), combines via OR,
    then applies morphological operations.
    
    Args:
        img: Input image (BGR)
        config: MaskingPipelineConfig instance
    
    Returns:
        Masked image (disease region revealed)
    """
    # Initialize empty mask
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Binarize each colorspace (convert once for efficiency)
    if config.binarize_rgb_channels:
        mask_rgb = _binarize_channels(img, config.binarize_rgb_channels, pickRGBChannel)
        cv2.bitwise_or(combined_mask, mask_rgb, dst=combined_mask)
    
    if config.binarize_hsv_channels:
        img_hsv = toHSV(img)
        mask_hsv = _binarize_channels(img_hsv, config.binarize_hsv_channels, pickHSVChannel)
        cv2.bitwise_or(combined_mask, mask_hsv, dst=combined_mask)
        del img_hsv, mask_hsv
    
    if config.binarize_lab_channels:
        img_lab = toLAB(img)
        mask_lab = _binarize_channels(img_lab, config.binarize_lab_channels, pickLABChannel)
        cv2.bitwise_or(combined_mask, mask_lab, dst=combined_mask)
        del img_lab, mask_lab
    
    # Apply morphological operations
    for operation_name, kernel_size, morph_type in config.morph_operations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        combined_mask = cv2.morphologyEx(combined_mask, morph_type, kernel)
    
    # Apply mask to image to reveal disease region
    return cv2.bitwise_and(img, img, mask=combined_mask)

def RunEdgeDetectionOnOne(img, config):
    if config.method == "sobel":
        img_gray = toBnW(img)
        img_sobel_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        img_sobel_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(img_sobel_x, img_sobel_y)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        del img_sobel_x, img_sobel_y
        return edges
    elif config.method == "canny":
        img_gray = toBnW(img)
        edges = cv2.Canny(img_gray, config.canny_thresholds[0], config.canny_thresholds[1])
        return edges
    else:
        raise ValueError(f"Unsupported edge detection method: {config.method}")

# ...existing code...
def RunCornerDetectionOnOne(img, config):
    img_gray = toBnW(img)
    
    if config.method == "harris":
        dst = cv2.cornerHarris(img_gray, config.harris_block_size, config.harris_ksize, config.harris_k)

        dilated = cv2.dilate(dst, None)
        local_max_mask = (dst == dilated) & (dst > 0.01 * dst.max())


        thresh_mask = (local_max_mask.astype(np.uint8)) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh_mask, connectivity=8
        )

        corners = [
            (int(round(cx)), int(round(cy)))
            for label_id, (cx, cy) in enumerate(centroids)
            if label_id != 0 
        ]
        return corners
    
    elif config.method == "shi-tomasi":
        corners = cv2.goodFeaturesToTrack(
            img_gray,
            maxCorners=config.shi_tomasi_max_corners,
            qualityLevel=config.shi_tomasi_quality_level,
            minDistance=config.shi_tomasi_min_distance,
            useHarrisDetector=config.shi_tomasi_use_harris_detector,
        )
        return [(int(x), int(y)) for [[x, y]] in corners] if corners is not None else []
    
    else:
        raise ValueError(f"Unsupported corner detection method: {config.method}")
# ...existing code...

def RunFeatureExtractionOnOne(img, config):
    feature_blocks = []
    dictionary = None  # Lazy load dictionary only if BoW is needed

    for method in config.methods:
        if method == "custom":
            f = extract_custom_fft_concat(img)
        elif method == "hog":
            f = extract_hog_features(img)
        elif method == "LBP":
            f = extract_lbp_features(img)
        elif method == "BoW":
            if dictionary is None:
                dictionary = download_default_dictionary()
            f = extract_bow_features(img, dictionary)
        else:
            raise ValueError(f"Unsupported feature extraction method: {method}")

        feature_blocks.append(np.asarray(f).ravel())
        del f

    if not feature_blocks:
        return np.array([], dtype=np.float32)

    return np.concatenate(feature_blocks, axis=0)



