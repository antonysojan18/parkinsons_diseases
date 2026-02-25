import cv2
import numpy as np
from PIL import Image, ImageFilter

binary = np.random.choice([0, 255], size=(256, 256)).astype(np.uint8)

# OpenCV blur
smoothed_cv2 = cv2.GaussianBlur(binary, (15, 15), 0)
_, ideal_cv2 = cv2.threshold(smoothed_cv2, 127, 255, cv2.THRESH_BINARY)
dev_cv2 = np.bitwise_xor(binary, ideal_cv2)
deviants_cv2 = np.sum(dev_cv2 > 0)

# PIL blur
pil_img = Image.fromarray(binary)
# sigma for ksize 15: 0.3 * ((15-1)*0.5 - 1) + 0.8 = 0.3 * 6 + 0.8 = 2.6
smoothed_pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=2.6))
smoothed_pil = np.array(smoothed_pil_img)
ideal_pil = np.where(smoothed_pil > 127, 255, 0).astype(np.uint8)
dev_pil = np.bitwise_xor(binary, ideal_pil)
deviants_pil = np.sum(dev_pil > 0)

print(f"Deviants CV2: {deviants_cv2}")
print(f"Deviants PIL: {deviants_pil}")
print(f"Diff: {abs(deviants_cv2 - deviants_pil)}")
