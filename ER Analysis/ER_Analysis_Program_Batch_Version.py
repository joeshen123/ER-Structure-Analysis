from ER_Analysis_Module import *
import os
import glob

root = tk.Tk()
root.withdraw()
root.directory = filedialog.askdirectory()
Name = root.directory

app = App()
app.mainloop()

camera = app.camera_choice
ER_label = app.ER_label_choice


os.chdir(Name)

if camera == "Nikon Confocal":
  hdf_filenames = glob.glob('*_1.hdf5' )

else:
    hdf_filenames = glob.glob('*.czi' )


print(hdf_filenames)
n = 1
for hdf in hdf_filenames:
   if camera == "Nikon Confocal":
  #Extract the channel that contains ER from the images
     f = h5py.File(hdf, 'r')
     ER_image = f['488 Channel'][:]

   else:
    img = AICSImage(hdf)
    ER_image = img.get_image_data("ZYX",S=0,C=1)
    ER_image = resize(ER_image,(ER_image.shape[0],512,512))
    ER_image = ER_image[np.newaxis,...]

   #Run the analysis program
   directory_name = hdf.split('.hdf5')[0]

   ER_image_pipeline = cell_segment(ER_image,Choice=ER_label,directory=directory_name)
   ER_image_pipeline.img_norm_smooth()

   ER_image_pipeline.threshold_time_combined()

   ER_image_pipeline.create_table_regions()

   ER_image_pipeline.Normalize()

   ER_image_pipeline.plotting()
   #ER_image_pipeline.Napari_Viewer()

   ER_image_pipeline.saving()

   print("Finish No. {0} imaging file out of total {1} files".format(str(n),str(len(hdf_filenames))))

   n+= 1


