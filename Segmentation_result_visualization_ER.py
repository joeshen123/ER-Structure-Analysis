import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import messagebox
import pandas as pd
import h5py
from matplotlib.figure import Figure
import numpy as np
from matplotlib.pyplot import cm
import napari
from matplotlib.backends.backend_qt5agg import FigureCanvas
from aicsimageio import AICSImage
from skimage.transform import resize

root = tk.Tk()
root.withdraw()

filez = filedialog.askopenfilenames(parent=root, title='Choose a file')

print(filez)
filename = filez[0]
seg_name = filez[1]

f = h5py.File(filename, 'r')

ER_image = f['488 Channel'][:]

#img = AICSImage(filename)
#ER_image = img.get_image_data("ZYX",S=0,C=1)
#ER_image = resize(ER_image,(ER_image.shape[0],512,512))
#ER_image = ER_image[np.newaxis,...]


f1 = h5py.File(seg_name, 'r')
try:
  ER_image = f1['Raw Image'][:]

except:
    print('No image, use raw image from nd2 file')
    pass

try:
    x_pixel, y_pixel, z_step = f1['voxel_info']

except:
    voxel_info = simpledialog.askstring("Question", "Please specify voxel info (z,y,z)")
    x_pixel, y_pixel, z_step = [float(voxel) for voxel in voxel_info.split(',')]


Mid_section = f1['Segmented Mid Section'][:]
seg_img = f1['Segmented Images'][:]
threshold = f1['thresholded image'][:]



viewer = napari.Viewer()
viewer.add_image(ER_image, name='Raw Image', colormap='green', scale=[1, z_step, y_pixel, x_pixel], blending='additive')
viewer.add_image(threshold, name='Thresholded', scale=[1, z_step, y_pixel, x_pixel], blending='additive')
viewer.add_labels(seg_img, name='Segmented Object ', scale=[1, z_step, y_pixel, x_pixel], blending='additive')
viewer.add_image(Mid_section, name='Mid_section ', scale=[1, z_step, y_pixel, x_pixel], blending='additive')

napari.run()