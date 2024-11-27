# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import glob
import cv2
import shutil
import traceback
import numpy as np


def convert(images_dir, output_dir, converter, gamma=0, k=0):
  image_files = glob.glob(images_dir + "/*.jpg")
  for image_file in image_files:
    basename = os.path.basename(image_file)
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, converter)

    output_file = os.path.join(output_dir, basename)
    if k>9:
      image = sharpen(image, k)
    if gamma>0:
       image = gamma_correction(image, gamma)
    cv2.imwrite(output_file, image)
    print("--- Saved {}".format(output_file))

def adjuster(images_dir, output_dir, alpha=1, beta=0):
  image_files = glob.glob(images_dir + "/*.jpg")
  for image_file in image_files:
    basename = os.path.basename(image_file)
    image = cv2.imread(image_file)
    image = adjust(image, alpha, beta)

    output_file = os.path.join(output_dir, basename)
    cv2.imwrite(output_file, image)
    print("--- Saved {}".format(output_file))    


def adjust(img, alpha=1, beta=0):
    table = alpha * np.arange(256) + beta
    table = np.clip(table, 0, 255).astype(np.uint8)

    return cv2.LUT(img, table)

def sharpen(img, k):
    if k > 0:
      kernel = np.array([[-k, -k, -k], 
                       [-k, 1+8*k, -k], 
                       [-k, -k, -k]])
      img = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    return img

def gamma_correction(img, gamma):
    table = (np.arange(256) / 255) ** gamma * 255
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)

if __name__ == "__main__":
  try:
     images_dir = "./mini_test/images"
     output_dir = "./xyz_images"
     if os.path.exists(output_dir):
       shutil.rmtree(output_dir)
     os.makedirs(output_dir)
     convert(images_dir, output_dir, cv2.COLOR_BGR2XYZ, gamma=0, k=0)
     #adjuster(images_dir, output_dir, alpha=1, beta=0.5)

  except:
    traceback.print_exc()
 
