import cv2
import numpy as np

def otsu_threshold(gray_img):
    hist, _ = np.histogram(gray_img.flatten(), bins=256, range=(0, 256))
    hist_norm = hist.astype(float) / hist.sum()
    
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    
    p1 = Q
    p2 = 1.0 - p1
    
    # Add small epsilon to avoid division by zero
    p1 = np.where(p1 == 0, np.finfo(float).eps, p1)
    p2 = np.where(p2 == 0, np.finfo(float).eps, p2)
    
    mean1 = (bins * hist_norm).cumsum() / p1
    mean2 = ((bins * hist_norm).sum() - mean1 * p1) / p2
    
    variance12 = p1 * p2 * (mean1 - mean2) ** 2
    
    # OpenCV's Otsu returns the first maximum index
    return np.argmax(variance12)

# Generate a random test image
img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

cv_thresh, cv_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
my_thresh = otsu_threshold(img)
my_bin = np.where(img <= my_thresh, 255, 0).astype(np.uint8)

print(f"CV2 threshold: {cv_thresh}, My threshold: {my_thresh}")
print(f"Diff in binary: {np.sum(cv_bin != my_bin)}")
