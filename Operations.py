import os

from matplotlib import image
import numpy as np
import cv2
import urllib.request

def toBnW(img):
    # converts an image to black and white
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def toHSV(img):
    # converts an image to HSV
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img

def toLAB(img):
    # converts an image to LAB
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img

def pickLABChannel(img, channel):
    # channel one of L, A, B
    if len(img.shape) != 3:
        return img
    channel = channel.upper()
    if channel == 'L':
        return img[:,:,0]
    elif channel == 'A':
        return img[:,:,1]
    elif channel == 'B':
        return img[:,:,2]

def pickRGBChannel(img,channel):
    # channel one of R, G, B
    if len(img.shape) != 3:
        return img
    channel = channel.upper()
    if channel == 'R':
        return img[:,:,2]
    elif channel == 'G':
        return img[:,:,1]
    elif channel == 'B':
        return img[:,:,0]

def pickHSVChannel(img,channel):
    # channel one of H, S, V
    if len(img.shape) != 3:
        return img  # If the image is not a color image, return it as is
    channel = channel.upper()
    if channel == 'H':
        return img[:,:,0]
    elif channel == 'S':
        return img[:,:,1]
    elif channel == 'V':
        return img[:,:,2]

def thresh(img,th):
    # returns a binarized thresholded image
    return (img > th).astype(np.uint8) * 255

def otsu_thresh(img):
    if img is None or img.size == 0:
        return 0

    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    prob = hist / hist.sum()

    w0 = np.cumsum(prob)
    w1 = 1.0 - w0

    cum_sum = np.cumsum(prob * np.arange(256))
    total_mean = cum_sum[-1]

    # Standard formula without division protection
    var = ((total_mean * w0 - cum_sum) ** 2) / (w0 * w1)

    return int(np.nanargmax(var))

def lab_or_hsv_thresh(img):
    # Convert to LAB and HSV
    lab_img = toLAB(img)
    hsv_img = toHSV(img)

    # Pick A channel from LAB and S channel from HSV
    a_channel = pickLABChannel(lab_img, 'A')
    s_channel = pickHSVChannel(hsv_img, 'S')
    del lab_img, hsv_img

    # Compute Otsu's threshold for both channels
    a_thresh = otsu_thresh(a_channel)
    s_thresh = otsu_thresh(s_channel)

    # Apply thresholding
    a_binary = thresh(a_channel, a_thresh)
    s_binary = thresh(s_channel, s_thresh)
    del a_channel, s_channel

    # Combine the two binary images using logical OR
    combined_binary = np.maximum(a_binary, s_binary)
    del a_binary, s_binary

    return combined_binary

def extract_custom_fft_concat(img, target_size=(128, 128), patch_size=(16, 16), thresholds=(5, 10)):
    # 1. Resize image to guarantee fixed patch count across the dataset
    img_resized = cv2.resize(img, target_size)
    
    h, w = target_size
    ph, pw = patch_size
    
    # 2. Pre-compute distance map centered at (8, 8)
    cy, cx = ph // 2, pw // 2
    y_idx, x_idx = np.indices((ph, pw))
    distances = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)
    
    low_mask  = distances < thresholds[0]
    high_mask = distances > thresholds[1]
    mid_mask  = ~(low_mask | high_mask)
    del y_idx, x_idx, distances
    
    patch_vectors = []
    
    # 3. Slide through patches sequentially
    for y in range(0, h - ph + 1, ph):
        for x in range(0, w - pw + 1, pw):
            patch = img_resized[y:y + ph, x:x + pw]
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            
            # FFT -> Shift -> Magnitude
            patchfft = np.fft.fft2(patch_hsv, axes=(0, 1))
            patchfft_shifted = np.fft.fftshift(patchfft, axes=(0, 1))
            patch_mag = np.abs(patchfft_shifted)
            del patch_hsv, patchfft, patchfft_shifted
            
            # Extract region magnitudes
            low_region  = patch_mag[low_mask]
            mid_region  = patch_mag[mid_mask]
            high_region = patch_mag[high_mask]
            
            # Means & Std Stds per channel (3 values each)
            low_mean,  low_std  = np.mean(low_region, axis=0),  np.std(low_region, axis=0)
            mid_mean,  mid_std  = np.mean(mid_region, axis=0),  np.std(mid_region, axis=0)
            high_mean, high_std = np.mean(high_region, axis=0), np.std(high_region, axis=0)
            del patch_mag, low_region, mid_region, high_region
            
            # 18-element patch vector
            patch_vec = np.concatenate([
                low_mean, low_std, 
                mid_mean, mid_std, 
                high_mean, high_std
            ])
            
            patch_vectors.append(patch_vec)
            
    # 4. Concatenate ALL patch vectors into a single 1D feature array
    # Output length for 128x128 image: 64 patches * 18 = 1,152 floats
    del img_resized, low_mask, high_mask, mid_mask
    return np.concatenate(patch_vectors)

def extract_hog_features(img, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    """
    Extracts HOG features using native OpenCV HOGDescriptor if available,
    falling back to vectorized NumPy/Sobel gradient binning.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Try native OpenCV HOGDescriptor
    if hasattr(cv2, 'HOGDescriptor'):
        win_w = (gray.shape[1] // cell_size[1]) * cell_size[1]
        win_h = (gray.shape[0] // cell_size[0]) * cell_size[0]
        
        if win_w > 0 and win_h > 0:
            gray_resized = cv2.resize(gray, (win_w, win_h))
            block_w = block_size[1] * cell_size[1]
            block_h = block_size[0] * cell_size[0]
            stride_w = cell_size[1]
            stride_h = cell_size[0]
            
            hog = cv2.HOGDescriptor(
                _winSize=(win_w, win_h),
                _blockSize=(block_w, block_h),
                _blockStride=(stride_w, stride_h),
                _cellSize=(cell_size[1], cell_size[0]),
                _nbins=nbins
            )
            feats = hog.compute(gray_resized)
            if feats is not None:
                return feats.flatten()

    # Vectorized Pure-Python/Sobel Fallback
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    del gx, gy
    angle = angle % 180.0

    cell_h, cell_w = cell_size
    n_cells_y = gray.shape[0] // cell_h
    n_cells_x = gray.shape[1] // cell_w

    if n_cells_y == 0 or n_cells_x == 0:
        return np.array([], dtype=np.float32)

    # Crop to exact cell grid boundaries
    mag = mag[:n_cells_y * cell_h, :n_cells_x * cell_w]
    angle = angle[:n_cells_y * cell_h, :n_cells_x * cell_w]

    bin_width = 180.0 / nbins
    bin_idx = (angle // bin_width).astype(int) % nbins
    del angle

    # Bin accumulation via vector reshaping
    mag_reshaped = mag.reshape(n_cells_y, cell_h, n_cells_x, cell_w).swapaxes(1, 2)
    bin_reshaped = bin_idx.reshape(n_cells_y, cell_h, n_cells_x, cell_w).swapaxes(1, 2)
    del mag, bin_idx

    histograms = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)
    for b in range(nbins):
        histograms[:, :, b] = np.sum(mag_reshaped * (bin_reshaped == b), axis=(2, 3))

    del mag_reshaped, bin_reshaped
    return histograms.flatten()


def extract_lbp_features(img, radius=1, n_points=8):
    """
    Fast, fully vectorized Local Binary Pattern (LBP) feature extractor.
    Replaces slow nested Python coordinate loops with array slicing.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Pad image to simplify boundary handling
    padded = np.pad(gray, pad_width=radius, mode='reflect').astype(np.int32)
    center = padded[radius:-radius, radius:-radius]
    
    lbp_code = np.zeros(center.shape, dtype=np.uint32)

    # Process all sampling points simultaneously via indexing
    for point in range(n_points):
        theta = 2.0 * np.pi * point / n_points
        dy = int(round(radius * np.sin(theta)))
        dx = int(round(radius * np.cos(theta)))
        
        # Extract neighbor sub-grid
        y_start = radius + dy
        y_end = y_start + center.shape[0]
        x_start = radius + dx
        x_end = x_start + center.shape[1]
        
        neighbor = padded[y_start:y_end, x_start:x_end]
        
        # Bitwise shift-and-add LBP encoding
        lbp_code |= ((neighbor >= center).astype(np.uint32) << point)
    
    del padded, center

    # Compute normalized histogram
    hist, _ = np.histogram(lbp_code.ravel(), bins=np.arange(0, 2**n_points + 1), density=True)
    return hist.astype(np.float32)


def download_default_dictionary(
    save_path="existing_codebook.yml",
    url="https://raw.githubusercontent.com/itlab-vision/opencv-samples-perf-analysis/master/data/bow_svm/vocabulary.yml"
):
    """
    Downloads and loads a pre-computed BoW vocabulary matrix.
    Generates a local synthetic vocabulary fallback if remote download fails.
    """
    if not os.path.exists(save_path):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response, open(save_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception:
            # Fallback matrix generation if download fails
            default_dict = np.random.randn(50, 128).astype(np.float32)
            fs = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
            fs.write("vocabulary", default_dict)
            fs.release()
            return default_dict

    fs = cv2.FileStorage(save_path, cv2.FILE_STORAGE_READ)
    
    # Check "vocabulary" node safely without calling .isMat()
    vocab_node = fs.getNode("vocabulary")
    if not vocab_node.empty():
        dictionary = vocab_node.mat()
    else:
        dictionary = fs.getFirstTopLevelNode().mat()
        
    fs.release()

    if dictionary is None:
        raise ValueError(f"Failed to load OpenCV vocabulary matrix from {save_path}")

    return dictionary.astype(np.float32)


def extract_bow_features(img, dictionary):
    """
    Extracts Bag-of-Words feature histogram using direct SIFT descriptor matching.
    Removes dependency on cv2.BOWImgDescriptorExtractor.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    del sift

    vocab_size = dictionary.shape[0]

    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocab_size, dtype=np.float32)

    # Nearest neighbor matching against the visual dictionary
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(descriptors.astype(np.float32), dictionary.astype(np.float32))
    del descriptors, matcher

    # Build and L1-normalize occurrence histogram
    histogram = np.zeros(vocab_size, dtype=np.float32)
    for match in matches:
        histogram[match.trainIdx] += 1.0

    total_count = np.sum(histogram)
    if total_count > 0:
        histogram /= total_count

    return histogram