import os
import cv2
import numpy as np

def read_images(path):
  filenames = os.listdir(path)

  images = []
  for filename in filenames:
    img = cv2.imread(os.path.join(path, filename),-1)
    if img.max() > 1:
      img = img / 255
    img = cv2.resize(img, (64, 64))
    img[img!=0] = 1
    images.append(img)

  images = np.array(images)

  return images

mask_images = read_images("/home/lizhe/paper/StableVideo/TokenFlow/data/wolf_maskrcnn")
print(mask_images.shape)
expand_mask_images = np.expand_dims(mask_images, axis=1)
print(expand_mask_images.shape)
expand_mask_images = np.repeat(expand_mask_images, 4, axis=1)
print(expand_mask_images.shape)
print(expand_mask_images[0][0] == mask_images[0])
np.save("/home/lizhe/paper/StableVideo/TokenFlow/data/wolf_mask.npy", expand_mask_images)
