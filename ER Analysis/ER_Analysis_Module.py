import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import math
import napari
from aicsimageio import AICSImage
#Use Allen Segmenter to segment ER
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, edge_preserving_smoothing_3d,image_smoothing_gaussian_slice_by_slice
from skimage.morphology import remove_small_objects  # function for post-processing (size filter)
from skimage import io
from skimage.transform import resize
from aicssegmentation.core.MO_threshold import MO
from skimage.measure import label, regionprops
import numpy as np
import h5py
from tqdm import tqdm
from colorama import Fore
from skimage.restoration import rolling_ball
from skimage.restoration import ellipsoid_kernel
from skimage.filters import sobel, scharr
from skimage.segmentation import watershed
from skimage.morphology import binary_opening, disk,ball
from aicssegmentation.core.utils import hole_filling
from skimage.exposure import rescale_intensity
from pandas import DataFrame
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
import customtkinter
import matplotlib.pyplot as plt
import warnings
from qtpy.QtWidgets import QApplication

warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)
warnings.simplefilter("ignore",FutureWarning)

#Class to make a GUI where user can choose which camera is used to take the images/movies, how is the ER labelled(It will
#be different analysis pipeline if it is labelled via EGFP-KDEL or Sec61B and finally whether the user wants to crop the images for the analysis
class MyRadiobuttonFrame(customtkinter.CTkFrame):
    def __init__(self, master, title, values):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.values = values
        self.title = title
        self.radiobuttons = []
        self.variable = customtkinter.StringVar(value="")

        self.title = customtkinter.CTkLabel(self, text=self.title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

        for i, value in enumerate(self.values):
            radiobutton = customtkinter.CTkRadioButton(self, text=value, value=value, variable=self.variable)
            radiobutton.grid(row=i + 1, column=0, padx=10, pady=(10, 0), sticky="w")
            self.radiobuttons.append(radiobutton)

    def get(self):
        return self.variable.get()

    def set(self, value):
        self.variable.set(value)


class MyCheckboxFrame(customtkinter.CTkFrame):
    def __init__(self, master, title, values):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.values = values
        self.title = title
        self.checkboxes = []

        self.title = customtkinter.CTkLabel(self, text=self.title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

        for i, value in enumerate(self.values):
            checkbox = customtkinter.CTkCheckBox(self, text=value)
            checkbox.grid(row=i + 1, column=0, padx=10, pady=(10, 0), sticky="w")
            self.checkboxes.append(checkbox)

    def get(self):
        checked_checkboxes = None
        for checkbox in self.checkboxes:
            if checkbox.get() == 1:
                checked_checkboxes = checkbox.cget("text")

        return checked_checkboxes


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.camera_choice = None
        self.ER_label_choice = None
        self.title("NM ER Analysis")
        self.geometry("1000x600")
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.radiobutton_frame1 = MyRadiobuttonFrame(self, "Which Camera is used for imaging?",
                                                     values=["Nikon Confocal", "Zeiss Elyra"])
        self.radiobutton_frame1.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        self.radiobutton_frame2 = MyRadiobuttonFrame(self, "How is ER labelled?", values=["EGFP-KDEL", "Sec-61B"])
        self.radiobutton_frame2.grid(row=0, column=1, padx=(0, 10), pady=(10, 0), sticky="nsew")
        self.checkbox_frame1 = MyCheckboxFrame(self, "Do you want to crop the movie?",
                                               values=["Yes, of course!", "No, skip it!"])
        self.checkbox_frame1.grid(row=0, column=2, padx=(0, 10), pady=(10, 0), sticky="nsew")

        self.button = customtkinter.CTkButton(self, text="Submit", command=self.button_callback)
        self.button.grid(row=3, column=0, padx=10, pady=10, sticky="ew", columnspan=3)

    def button_callback(self):
        self.camera_choice = self.radiobutton_frame1.get()
        self.ER_label_choice = self.radiobutton_frame2.get()
        self.cropping = self.checkbox_frame1.get()
        self.destroy()
        self.quit()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()
            self.quit()
            print('Program Exit by User Input!')
            quit()


from skimage.segmentation import clear_border
## Make the 3D Cell Segmentation Pipeline

# Initiate the cell segmentation class to store  segmentation result
from skimage import filters


class cell_segment:
    def __init__(self, Time_lapse_image, norm_factor=None, sigma=1, scale=0.22, Choice=None, scale_z=1.3,
                 directory=None):
        if norm_factor is None:
            self.norm_factor = [2.5, 7.5]

        self.image = Time_lapse_image.copy()
        self.Time_pts = Time_lapse_image.shape[0]

        self.smooth_param = sigma

        self.bw_img = np.zeros(self.image.shape)
        self.bw_midplane = np.zeros(self.image.shape, dtype=np.uint8)
        self.struct_image = np.zeros(self.image.shape)
        self.structure_img_smooth = np.zeros(self.image.shape)
        self.segmented_object_image = np.zeros(self.image.shape, dtype=np.uint8)
        self.seed_map_list = np.zeros(self.image.shape, dtype=np.uint8)
        self.ER_area = []
        self.ER_solidity = []
        self.scale = scale
        self.scale_z = scale_z
        self.choice = Choice
        self.static_canvas = None
        self.frame_list = []
        self.norm_area_list = []
        self.norm_perimeter_list = []
        self.norm_circularity_list = []
        self.directory = directory
        self.thin_dist_preserve = 3.4
        self.thin_dist = 2

    # Make a function to get middle frame based on segmented area
    @staticmethod
    def get_middle_frame_area(labelled_image_stack, choice='number'):
        max_area = 0
        max_label_num = 0
        max_n = 0
        max_circularity = 0
        for z in range(labelled_image_stack.shape[0]):
            img_slice = labelled_image_stack[z, :, :]

            if choice == 'number':
              _,label_num = label(img_slice,connectivity=1, return_num=True)

              if label_num >= max_label_num:
                 max_label_num = label_num
                 max_n = z

            elif choice == 'area':
              area = np.count_nonzero(img_slice)
              #print(area,z)
              if area >= max_area:
                 max_area = area
                 max_n = z

        #print (max_n)

        return max_n

    # Make a function to retrieve the seed for watershed segmentation algorithm
    @staticmethod
    def get_3dseed_from_mid_frame(bw_1, stack_shape, mid_frame, hole_min, bg_seed=True):
        from skimage.morphology import remove_small_objects
        out = remove_small_objects(bw_1 > 0, hole_min)

        out1 = label(out)
        stat = regionprops(out1)

        # build the seed for watershed
        seed = np.zeros(stack_shape)
        seed_count = 0
        if bg_seed:
            seed[0, :, :] = 1
            seed_count += 1

        for idx in range(len(stat)):
            py, px = np.round(stat[idx].centroid)
            seed_count += 1
            seed[mid_frame, int(py), int(px)] = seed_count

        return seed

    # define function to crop the images
    def cropping_image (self):


        viewer = napari.Viewer()
        viewer.add_image(self.image, blending='additive', name="Raw Images", colormap='green', visible=True,
                         scale=(1, self.scale_z, self.scale, self.scale))
        rec_layer = viewer.add_shapes(ndim=4, shape_type='rectangle', edge_width=1.5,
                                       edge_color='b')

        napari.run()

        y1, _, y2, _ = rec_layer.data[0][:, 2]
        x1, x2, _, _ = rec_layer.data[0][:, 3]
        y1 = int(y1/self.scale)
        y2 = int(y2/self.scale)

        x1 = int(x1/self.scale)
        x2 = int(x2/self.scale)

        if y1 <=0:
            y1 = 0

        if y1 >= self.image.shape[2]:
            y1 = self.image.shape[2]

        if y2 <= 0:
            y2 = 0

        if y2 >= self.image.shape[2]:
            y2 = self.image.shape[2]

        if x1 <= 0:
            x1 = 0

        if x1 >= self.image.shape[3]:
            x1 = self.image.shape[3]

        if x2 <= 0:
            x2 = 0

        if x2 >= self.image.shape[3]:
            x2 = self.image.shape[3]

        temp_image = np.zeros(self.image.shape)
        temp_image[:, :, y1:y2, x1:x2] = self.image[:, :, y1:y2, x1:x2]

        print(self.image[:, :, y1:y2, x1:x2].shape)
        self.image = temp_image

    # define a function to apply normalization and smooth on Time lapse images
    def img_norm_smooth(self):
        pb = tqdm(range(self.Time_pts), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET))
        for t in pb:
            pb.set_description("Smooth and background substraction")
            img = self.image[t].copy()

            # Clip data and remove extreme bright speckles
            #vmin, vmax = np.percentile(img, q=(0.5, 99.5))

            #clipped_img = rescale_intensity(img, in_range=(vmin, vmax), out_range=np.uint16)

            # self.structure_img_smooth[t] = edge_preserving_smoothing_3d(img, numberOfIterations=5)
            self.struct_image[t] = intensity_normalization(img, scaling_param=self.norm_factor)

            if self.choice == 'EGFP-KDEL':
                self.structure_img_smooth[t] = image_smoothing_gaussian_slice_by_slice(self.struct_image[t],
                                                                                       sigma=self.smooth_param)

            else:
                self.structure_img_smooth[t] = edge_preserving_smoothing_3d(self.struct_image[t])
                #self.structure_img_smooth[t] = image_smoothing_gaussian_slice_by_slice(self.struct_image[t],
                #                                                                       sigma=2)
            # Use rolling_ball to remove background in Z direction
            background = rolling_ball(self.structure_img_smooth[t], kernel=ellipsoid_kernel((20, 1, 1), 0.2))
            self.structure_img_smooth[t] = self.structure_img_smooth[t] - background

    # define a function to apply Ostu Object thresholding followed by seed-based watershed to each time point
    def threshold_Time_KDEL(self):
        pb = tqdm(range(self.Time_pts), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
        for t in pb:
            pb.set_description("Thresholding and Watershed Segmentation")
            # MO threhodling
            # Check if there is NAN data
            img_sum = np.sum(self.structure_img_smooth[t])
            Nan_check = np.isnan(img_sum)

            # If there is NAN in the data, skip analysis for this image
            if Nan_check == True:
                print("Nan image found")
                continue

            bw, object_for_debug = MO(self.structure_img_smooth[t], global_thresh_method='med', extra_criteria=True,
                                      object_minArea=100, return_object=True)

            # Morphological operations to fill holes and remove small/touching objects
            bw = binary_opening(bw, selem=np.ones((1, 1, 1)))

            bw = hole_filling(bw, 1, 5, fill_2d=True)
            bw = remove_small_objects(bw > 0, min_size=50, connectivity=1, in_place=False)

            self.bw_img[t] = bw

            # Get middle frame
            if t == 0:
             mid_z = cell_segment.get_middle_frame_area(bw,choice='area')
             #print(mid_z)

            else:
                mid_z_temp = cell_segment.get_middle_frame_area(bw,choice='area')

                if abs(mid_z_temp - mid_z) < 4:
                    mid_z = mid_z_temp

            bw_mid_z = bw[mid_z, :, :]
            bw_mid_z = remove_small_objects(bw_mid_z > 0, min_size=30, connectivity=1, in_place=False)

            self.bw_midplane[t, mid_z, :, :] = label(bw_mid_z)

            # Get seed map
            seed = cell_segment.get_3dseed_from_mid_frame(bw_mid_z, bw.shape, mid_z, 0, bg_seed=False)

            edge = scharr(self.image[t])
            seg = watershed(edge, markers=label(seed), mask=bw, watershed_line=True)

            seg = remove_small_objects(seg > 0, min_size=200, connectivity=1, in_place=False)
            seg = hole_filling(seg, 1, 20, fill_2d=True)
            final_seg = label(seg)

            self.segmented_object_image[t] = final_seg
            self.seed_map_list[t] = seed

    # define a function to apply filament segmentation if ER is labelled in Sec61B
    def threshold_Time_Sec61B(self):
        pb = tqdm(range(self.Time_pts), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
        for t in pb:
            pb.set_description("Using Filament 2D analysis pipeline")
            # MO threhodling
            # Check if there is NAN data
            img_sum = np.sum(self.structure_img_smooth[t])
            Nan_check = np.isnan(img_sum)

            # If there is NAN in the data, skip analysis for this image
            if Nan_check == True:
                print("Nan image found")
                continue

            f2_param = [[1.6,0.2]]


            bw = filament_2d_wrapper(self.structure_img_smooth[t], f2_param)

            # Morphological operations to fill holes and remove small/touching objects
            bw = binary_opening(bw, selem=np.ones((1, 1, 1)))
            bw = hole_filling(bw, 1, 300, fill_2d=True)
            bw = remove_small_objects(bw > 0, min_size=300, connectivity=1, in_place=False)

            self.bw_img[t] = bw
            self.segmented_object_image[t] = label(bw)

            if t == 0:
                mid_z = cell_segment.get_middle_frame_area(bw, choice='area')
                # print(mid_z)

            else:
                mid_z_temp = cell_segment.get_middle_frame_area(bw, choice='area')

                if abs(mid_z_temp - mid_z) < 4:
                      mid_z = mid_z_temp
            print(mid_z)
            bw_mid_z = bw[mid_z, :, :]
            bw_mid_z = remove_small_objects(bw_mid_z > 0, min_size=50, connectivity=1, in_place=False)

            self.bw_midplane[t, mid_z, :, :] = label(bw_mid_z)

    def threshold_time_combined(self):

        if self.choice == "EGFP-KDEL":
            self.threshold_Time_KDEL()

        else:
            self.threshold_Time_Sec61B()

    # function to create pandas table of cell attributes without tracking info
    def create_table_regions(self):

        positions = []
        pb = tqdm(range(self.image.shape[0]), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET))
        for n in pb:
            pb.set_description("Create table")
            labelled_slice = np.max(self.bw_midplane[n], axis=0)

            area_list = []
            for r in regionprops(labelled_slice):
                area_list.append(r.area)

            if not area_list:
               continue


            min_area = np.percentile(area_list, 40, axis=0)

            num = 1
            for region in regionprops(labelled_slice):

                if region.area > min_area:

                    position = []

                    y_row = region.centroid[0]
                    x_col = region.centroid[1]

                    ER_area = region.area * pow((self.scale), 2)
                    ER_perimeter = region.perimeter * (self.scale)
                    ER_Circularity = 4 * math.pi * ER_area / (ER_perimeter * ER_perimeter)

                    position.append(x_col)
                    position.append(y_row)

                    position.append(int(n))
                    position.append(num)
                    position.append(ER_area)
                    position.append(ER_perimeter)
                    position.append(ER_Circularity)
                    num += 1

                else:
                    continue

                positions.append(position)

            area_list_temp = area_list
        self.positions_table = DataFrame(positions, columns=['x', 'y', "frame", 'object number', 'area', 'perimeter',
                                                             'circularity'])

    # Functions to normalize ER area, perimeter and circularity
    def Normalize(self):

        self.positions_table['normalized area'] = self.positions_table['area']
        self.positions_table['normalized perimeter'] = self.positions_table['perimeter']
        self.positions_table['normalized circularity'] = self.positions_table['circularity']

        area_norm = np.max(self.positions_table.loc[self.positions_table.frame == 0, "area"])
        perimeter_norm = np.max(self.positions_table.loc[self.positions_table.frame == 0, "perimeter"])
        circularity_norm = np.mean(self.positions_table.loc[self.positions_table.frame == 0, "circularity"])

        self.positions_table['normalized area'] = self.positions_table['normalized area'] / area_norm
        self.positions_table['normalized perimeter'] = self.positions_table['normalized perimeter'] / perimeter_norm
        self.positions_table['normalized circularity'] = self.positions_table['normalized circularity'] / circularity_norm

    # Plot change of normalized ER area, ER perimeter and circularity of ER network
    def plotting(self, napari_integration=True):

        if napari_integration == True:
            static_canvas = FigureCanvas(Figure(figsize=(3, 1)))

            axes = static_canvas.figure.subplots(3, sharex=True)

            # list all remaining tracks

            axes[0].set_ylabel('Connected ER area change (norm.)', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Connected ER perimeter change (norm.)', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Time/Frame', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('ER circularity change (norm.)', fontsize=14, fontweight='bold')

            color = cm.tab20b(np.linspace(0, 1, 3))

            for t in range(self.image.shape[0]):
                df_subset = self.positions_table[self.positions_table.frame == t]
                area = np.max(df_subset['normalized area'])
                perimeter = np.max(df_subset['normalized perimeter'])
                circularity = np.mean(df_subset['normalized circularity'])

                self.frame_list.append(t)
                self.norm_area_list.append(area)
                self.norm_perimeter_list.append(perimeter)
                self.norm_circularity_list.append(circularity)

            axes[0].plot(self.frame_list, self.norm_area_list, color=color[0], marker='o')
            axes[1].plot(self.frame_list, self.norm_perimeter_list, color=color[1], marker='o')
            axes[2].plot(self.frame_list, self.norm_circularity_list, color=color[2], marker='o')

            axes[0].tick_params(axis='y', labelsize=15)
            axes[1].tick_params(axis='y', labelsize=15)
            axes[2].tick_params(axis='both', labelsize=15)

            self.static_canvas = static_canvas

        else:
            fig, axes = plt.subplots(3, figsize=(3, 1), sharex=True)
            fig.set_size_inches(14, 18)

            color = cm.tab20b(np.linspace(0, 1, 3))

            axes[0].set_ylabel('Connected ER area change (norm.)', fontsize=16, fontweight='bold')
            axes[1].set_ylabel('Connected ER perimeter change (norm.)', fontsize=16, fontweight='bold')
            axes[2].set_xlabel('Time/Frame', fontsize=16, fontweight='bold')
            axes[2].set_ylabel('ER circularity change (norm.)', fontsize=16, fontweight='bold')

            axes[0].plot(self.frame_list, self.norm_area_list, color=color[0], marker='o')
            axes[1].plot(self.frame_list, self.norm_perimeter_list, color=color[1], marker='o')
            axes[2].plot(self.frame_list, self.norm_circularity_list, color=color[2], marker='o')

            axes[0].tick_params(axis='y', labelsize=15)
            axes[1].tick_params(axis='y', labelsize=15)
            axes[2].tick_params(axis='both', labelsize=15)

            fig_save_name = '{File_Name}_analysis plot.png'.format(File_Name=self.directory)
            plt.savefig(fig_save_name, dpi=100)

    # Functions to save analysis dataframe, segmentation result and analysis plot for downstream pipeline
    def saving(self):

        seg_save_name = '{File_Name}_segmentation_result.hdf5'.format(File_Name=self.directory)

        with h5py.File(seg_save_name, "w") as f:
            f.create_dataset('Raw Image', data=self.image, compression='gzip')
            f.create_dataset('Segmented Mid Section', data=self.bw_midplane, compression='gzip')
            f.create_dataset('Segmented Images', data=self.segmented_object_image, compression='gzip')
            f.create_dataset('thresholded image', data=self.bw_img, compression='gzip')

            if self.choice == 'EGFP-KDEL':
                f.create_dataset('Seed Map', data=self.seed_map_list, compression='gzip')

        table_save_name = '{File_Name}_result.csv'.format(File_Name=self.directory)

        self.positions_table.to_csv(table_save_name, index=False)

        self.plotting(napari_integration=False)

    # Functions to visualize the segmentation results in Napari
    def Napari_Viewer(self):

        if self.choice == "EGFP-KDEL":

           viewer = napari.Viewer()
           viewer.add_image(self.image, blending='additive', name="Raw Images", colormap='green',visible=False,
                             scale=(1, self.scale_z, self.scale, self.scale))
           viewer.add_image(self.structure_img_smooth, blending='additive', name="Smooth Images", colormap='gray',visible=False,
                             scale=(1, self.scale_z, self.scale, self.scale))
           viewer.add_image(self.bw_img, colormap='cyan', blending='additive', name="Thresholded Images",visible=False,
                             scale=(1, self.scale_z, self.scale, self.scale))
           peaks = np.nonzero(self.seed_map_list)
           viewer.add_points(np.array(peaks).T, name='peaks', size=5, face_color='red',
                              scale=(1, self.scale_z, self.scale, self.scale), blending='additive', visible=False)
           viewer.add_labels(self.segmented_object_image, name="Segmented Images", blending="additive",visible=False,
                              scale=(1, self.scale_z, self.scale, self.scale))
           viewer.add_labels(self.bw_midplane, blending="additive", name="Segmented Mid Section",
                              scale=(1, self.scale_z, self.scale, self.scale))

           # Plotting
           viewer.window.add_dock_widget(self.static_canvas, area='right', name='Analysis Plot')

           napari.run()

        else:
            viewer = napari.Viewer()
            viewer.add_image(self.image, blending='additive', name="Raw Images", colormap='green',visible=False,
                             scale=(1, self.scale_z, self.scale, self.scale))
            viewer.add_image(self.structure_img_smooth, blending='additive', name="Smooth Images", colormap='gray',visible=False,
                             scale=(1, self.scale_z, self.scale, self.scale))
            viewer.add_image(self.bw_img, colormap='cyan', blending='additive', name="Thresholded Images",visible=False,
                             scale=(1, self.scale_z, self.scale, self.scale))
            viewer.add_labels(self.segmented_object_image, name="Segemnted Images", blending="additive",visible=False,
                              scale=(1, self.scale_z, self.scale, self.scale))
            viewer.add_labels(self.bw_midplane, blending="additive", name="Segmented Mid Section",
                              scale=(1, self.scale_z, self.scale, self.scale))

            # Plotting
            viewer.window.add_dock_widget(self.static_canvas, area='right', name='Analysis Plot')

            napari.run()
