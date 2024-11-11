from ER_Analysis_Module import *

# Import a series of timelapse image into Python program
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Image files', ['.hdf5','.czi'])]

filename = filedialog.askopenfilenames(parent = root, title='Please Select a File', filetypes = my_filetypes)[0]
print(filename)

app = App()
#Finish the program is the user hits the 'X' button
app.protocol("WM_DELETE_WINDOW", app.on_closing)
app.mainloop()
camera = app.camera_choice
ER_label = app.ER_label_choice
cropping = app.cropping
directory_name = filename.split('.hdf5')[0]


if camera == "Nikon Confocal":
#Extract the channel that contains ER from the images
  f = h5py.File(filename, 'r')
  ER_image = f['488 Channel'][:]

else:
  img = AICSImage(filename)

  ER_image = img.get_image_data("ZYX",S=0,C=1)
  ER_image = resize(ER_image,(ER_image.shape[0],512,512))
  ER_image = ER_image[np.newaxis,...]


#Run the analysis program
ER_image_pipeline = cell_segment(ER_image,Choice=ER_label,directory=directory_name)

if 'Yes' in cropping:
  ER_image_pipeline.cropping_image()

ER_image_pipeline.img_norm_smooth()

ER_image_pipeline.threshold_time_combined()

ER_image_pipeline.create_table_regions()

ER_image_pipeline.Normalize()

ER_image_pipeline.plotting()

ER_image_pipeline.Napari_Viewer()

ER_image_pipeline.saving()

