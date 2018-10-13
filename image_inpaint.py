from image_processing import ImageProcessor
from network_settings import *
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from network import Network

import os
import tkinter as tk
import tensorflow as tf
import numpy as np


class ImageInpaint:
    """
    ImageInpaint is responsible for providing a front-end user interface, which provides access to the following tasks:
        - Browse and choose an image from their computer
        - Select a region to be complete
        - Invoke the generator to complete the image
        - Save the completed image to their computer
    """

    image_processor = ImageProcessor()
    start_x = 0
    start_y = 0
    ratio = 4
    canvas_max_size = 512
    selection_box_width = patch_width * ratio
    patch_width_absolute = patch_width
    padding = patch_width * ratio

    selection_coordinates = []

    selection_visible = False
    original_image_visible = False
    faces_model_enabled = False

    completed_image = None
    selection = None
    sess = None
    saver = None
    g_input = None
    g_output_patch_only = None
    surrounding_region = None
    training = None
    img = None
    original_image_resized = None
    original_image = None
    image_height = None
    image_width = None
    unsharp_mask_slider = None
    last_generated_patch = None
    last_masked_image = None
    last_patch_start_x = None
    last_patch_start_y = None


    def __init__(self):
        """
        Initialises the window and controls, loads the model, and starts the main loop

        """

        # Create the window and canvas__________________________________________________________________________________

        self.window = tk.Tk()
        self.window.title("Image Inpainter")
        self.window.geometry("800x600")
        self.window.configure(background='white')

        self.canvas = Canvas(self.window, width=1, height=1, borderwidth=0, bd=0, highlightthickness=0, relief='ridge')

        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<Button-1>", self.mouse_down)

        # Add containers to the window which hold the controls and image________________________________________________

        controls_row_1 = Frame(self.window)
        controls_row_2 = Frame(self.window)
        image_holder = Frame(self.window)

        controls_row_1.pack(side=TOP)
        controls_row_2.pack(side=TOP, pady=10)
        image_holder.pack(side=BOTTOM, pady=10)

        # Add controls__________________________________________________________________________________________________

        button_width = 10

        self.open_image_button = Button(self.window, text="Open", command=self.open_image, width=button_width)

        self.selection_button = Button(self.window, text="Select", command=self.toggle_selection, width=button_width,
                                       state=DISABLED)

        self.complete_button = Button(self.window, text="Complete", command=self.complete_image, width=button_width,
                                      state=DISABLED)

        self.save_button = Button(self.window, text="Save", command=self.save_image, width=button_width,
                                  state=DISABLED)

        self.toggle_original_button = Button(self.window, text="Ground Truth", command=self.toggle_original_image,
                                             width=button_width, state=DISABLED)

        self.switch_model_button = Button(self.window, text="Faces Model", command=self.switch_model,
                                             width=button_width)

        self.unsharp_mask_slider = Scale(self.window, from_=0, to_=100, orient=HORIZONTAL, bg='white', bd=1,
                                         troughcolor='white', activebackground='#e7e7e7', length=150, width=10,
                                         command=self.unsharp_mask, showvalue=0, state=DISABLED)

        self.unsharp_mask_label = Label(self.window, text="Unsharp Mask: -%")

        self.open_image_button.pack(in_=controls_row_1, side=LEFT)
        self.save_button.pack(in_=controls_row_1, side=LEFT)
        self.selection_button.pack(in_=controls_row_1, side=LEFT)
        self.complete_button.pack(in_=controls_row_1, side=LEFT)
        self.toggle_original_button.pack(in_=controls_row_1, side=LEFT)
        self.switch_model_button.pack(in_=controls_row_1, side=LEFT)

        self.unsharp_mask_label.pack(in_=controls_row_2, side=LEFT, padx=2)
        self.unsharp_mask_slider.pack(in_=controls_row_2, side=LEFT, padx=5)

        # Load model and start main loop________________________________________________________________________________
        self.setup_network()
        self.load_model(model_path)
        self.window.mainloop()


    def switch_model(self):
        """
        Switches between the generic model and the one trained on just faces
        """
        self.faces_model_enabled = not self.faces_model_enabled

        if self.faces_model_enabled:
            self.load_model(faces_model_path)
            self.switch_model_button.config(text="Generic Model")
        else:
            self.load_model(model_path)
            self.switch_model_button.config(text="Faces Model")


    def unsharp_mask(self, strength):
        """
        Sharpens the generated patch
        :param
            strength:   Controls the amount the image is sharpened by. The greater the value, the more sharp it becomes.
                        A value of 0 makes no change to the image

        """
        strength = int(strength)
        self.unsharp_mask_label.config(text="Unsharp Mask: {:d}%".format(strength))

        if self.last_generated_patch is None:
            return

        strength /= 20

        img = self.image_processor.unsharp_mask(self.last_generated_patch, strength)

        img = self.image_processor.merge_patch_with_image(img, self.last_masked_image,
                                                          self.last_patch_start_x, self.last_patch_start_y)

        self.img = img.astype('uint8')
        img = Image.fromarray(self.img, 'RGB')
        img = self.resize_image(img)
        self.completed_image = ImageTk.PhotoImage(img)
        self.display_image(self.completed_image)


    def open_image(self):
        """
        Displays the browse file dialog and, upon opening an image, displays the image and enables the remaining buttons
        """

        # Present a file dialog window, allowing only jpg and png files to be selected
        file = filedialog.askopenfile(parent=self.window, mode='rb', title='Select Image',
                                      filetypes=[('Jpeg', '*.jpeg'), ('Jpg', '*.jpg'), ('png', '*.png')])

        self.unsharp_mask_slider.config(state=DISABLED)

        if file is not None:

            img = Image.open(file)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # This copy is used by the network to complete the image
            self.img = np.array(img)
            # self.img copy will be wrote over when a user makes a modification, allowing multiple changes to a
            # a single image. self.original_image remains the same for the entire time the image is still loaded into
            # the program
            self.original_image = np.array(img)

            # Resize the image to visually fit the visible canvas
            img = self.resize_image(img)

            # Convert to a tkinter image and display it in the window
            self.completed_image = ImageTk.PhotoImage(img)
            self.original_image_resized = self.completed_image
            self.display_image(self.completed_image)

            self.original_image_visible = True
            self.selection_visible = False
            self.enable_buttons()


    def enable_buttons(self):
        """
        When the program is first launched, no image is loaded, therefore the following buttons are initially disabled.
        Upon importing an image, this function should be called to enable them
        """
        self.selection_button.config(state=NORMAL)
        self.complete_button.config(state=NORMAL)
        self.toggle_original_button.config(state=NORMAL)
        self.save_button.config(state=NORMAL)


    def save_image(self):
        """
        Presents the user with a save dialog allowing them to save their modified image
        """
        file = filedialog.asksaveasfile(mode='wb', defaultextension=".png")
        # Check a file has successfully been opened, and whether the user has the original image displayed or the
        # generated one and save the relevant one
        if file:
            if self.original_image_visible:
                Image.fromarray(self.original_image).save(file)
            else:
                Image.fromarray(self.img).save(file)


    def toggle_original_image(self):
        """
        Switch between displaying the original image or the one being modified
        """

        self.selection_visible = False
        self.canvas.itemconfig(self.selection, state='hidden')

        if self.original_image_visible:
            self.display_image(self.completed_image)
            self.original_image_visible = False
            self.toggle_original_button.config(text="Completed")
        else:
            self.display_image(self.original_image_resized)
            self.original_image_visible = True
            self.toggle_original_button.config(text="Ground Truth")


    def toggle_selection(self):
        """
        Toggle the visibility of the selection box
        """
        if self.selection_visible:
            self.canvas.itemconfig(self.selection, state='hidden')
            self.selection_visible = False
        else:
            self.canvas.itemconfig(self.selection, state='normal')
            self.selection_visible = True


    def display_image(self, img):
        """
        Displays the image in the window
        :argument
            img:    Image to be displayed in the window
        """

        self.canvas.config(width=img.width(), height=img.height())

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=img, anchor="nw")

        self.canvas.pack()
        self.add_selection_box()


    def add_selection_box(self):
        """
        Draws a selection box which the user can move around the image to choose the region they wish to complete
        """
        self.selection = self.canvas.create_rectangle(self.image_width // 2 - self.selection_box_width // 2,
                                                      self.image_height // 2 - self.selection_box_width // 2,
                                                      self.image_width // 2 + self.selection_box_width // 2,
                                                      self.image_height // 2 + self.selection_box_width // 2,
                                                      fill='black', width=2, state='hidden')

    def setup_network(self):
        """
        Setup the network tensors
            - g_input: Input to the generator
            - g_output_patch_only: Patch generated
            - surrounding_region: Region surrounding the masked image to be merged with the generated patch
            - training: Whether the model is training or not. When invoking the model, False should be passed in
        """
        network = Network()
        d_input, g_input, g_output, g_output_patch_only, d_optimizer, g_optimizer, surrounding_region, \
            patch_ground_truth, d_cost_fake, d_cost_real, g_cost, training = network.network()

        # Create a new TensorFlow session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

        self.g_input = g_input
        self.g_output_patch_only = g_output_patch_only
        self.surrounding_region = surrounding_region
        self.training = training


    def load_model(self, current_model_path):
        """
        Load the learnt model
        :param
            current_model_path:     Path to the learnt model

        """
        self.open_image_button.config(state=DISABLED)
        self.complete_button.config(state=DISABLED)

        # If the model is not successfully restored, disable the browse button to prevent attempts to invoke the network
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(current_model_path))
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            self.open_image_button.config(state=NORMAL)
            self.complete_button.config(state=NORMAL)
            print("Model Restored")
        else:
            print("WARNING: Model not restored")



    def complete_image(self):
        """
        Completes the selected region of the image and updates the visible image to reflect the changes
        """
        image_processor = ImageProcessor()

        # Convert the visible coordinates to actual pixel coordinates
        selection_coordinates = self.canvas.coords(self.selection)
        patch_start_x = int(selection_coordinates[0] // self.ratio)
        patch_start_y = int(selection_coordinates[1] // self.ratio)

        # Get the image components required to generate the patch and insert it back into the original
        g, masked_image, surrounding_region = image_processor.create_image_components(self.img, patch_start_x,
                                                                                      patch_start_y)

        # Generated the patch
        generated_patch = self.sess.run(self.g_output_patch_only, feed_dict={self.g_input: g,
                                                  self.surrounding_region: surrounding_region, self.training: False})

        # Store the last generated patch details to allow quick adjustments to the sharpness
        self.last_generated_patch = generated_patch[0]
        self.last_masked_image = masked_image
        self.last_patch_start_x = patch_start_x
        self.last_patch_start_y = patch_start_y

        # Sharpen generated patch and merge back into original
        generated_patch = image_processor.unsharp_mask(self.last_generated_patch)
        img = image_processor.merge_patch_with_image(generated_patch, masked_image, patch_start_x, patch_start_y)

        # View the complete image and set state of relevant controls
        self.img = img.astype('uint8')
        img = Image.fromarray(self.img, 'RGB')
        img = self.resize_image(img)
        self.completed_image = ImageTk.PhotoImage(img)
        self.display_image(self.completed_image)

        self.unsharp_mask_slider.config(state=NORMAL)
        self.unsharp_mask_slider.set(50)
        self.selection_visible = False
        self.original_image_visible = False


    def resize_image(self, img):
        """
        Resize the image to be displayed to the user. NOTE: This does not resize the image being completed or saved, but
        is rather just to fill the visible window
        :argument
            img:    Image to be resized
        """

        # Check which dimension is the maximum and fill the window along that dimension. Need to calculate the ratio if
        # a non-square image is loaded and resize the smaller side appropriately
        image_height = float(img.height)
        image_width = float(img.width)
        max_dimen = max(image_height, image_width)

        if max_dimen == image_height:
            ratio = float(self.canvas_max_size) / image_height
            image_height = float(self.canvas_max_size)
            image_width *= ratio
        else:
            ratio = float(self.canvas_max_size) / image_width
            image_width = float(self.canvas_max_size)
            image_height *= ratio

        # Store the ratio of the original image with respect to the size of the visible canvas and adjust the size of
        # the selection box
        self.ratio = ratio
        self.selection_box_width = self.patch_width_absolute * ratio
        self.padding = patch_width * ratio

        self.image_height = int(image_height)
        self.image_width = int(image_width)

        return img.resize((self.image_width, self.image_height))

    def mouse_down(self, event):
        """
        On mouse down, store the position of the current selection and the point where the user has clicked.
        This is required to calculate the new position by mouse_move
        :argument
            event:  Contains information about the mouse event such as location

        """
        self.selection_coordinates = self.canvas.coords(self.selection)
        self.start_x = event.x
        self.start_y = event.y


    def mouse_move(self, event):
        """
        Move the selection box to the current mouse position on drag.
        The conditional checks are required to ensure the selection box does not go out of bounds, which is 16 pixels
        from any edge since this is a requirement of the neural network itself
        :argument
            event:  Contains information about the mouse event such as location

        """

        diff_x = self.start_x - event.x
        diff_y = self.start_y - event.y

        # Calculate the new locations for the four corners of the selection box
        start_x_new = self.selection_coordinates[0] - diff_x
        start_y_new = self.selection_coordinates[1] - diff_y
        end_x_new = self.selection_coordinates[2] - diff_x
        end_y_new = self.selection_coordinates[3] - diff_y

        # Ensure that the selection box does not leave the bounds of the image. This should leave a margin of pixels
        # surrounding the patch which the network uses to complete the masked out region
        if start_x_new < self.padding:
            start_x_new = self.padding
            end_x_new = start_x_new + self.selection_box_width

        if start_y_new < self.selection_box_width:
            start_y_new = self.padding
            end_y_new = start_y_new + self.selection_box_width

        if end_x_new > self.image_width - self.padding:
            start_x_new = self.image_width - self.selection_box_width - self.padding
            end_x_new = start_x_new + self.selection_box_width

        if end_y_new > self.image_height - self.padding:
            start_y_new = self.image_height - self.selection_box_width - self.padding
            end_y_new = start_y_new + self.selection_box_width

        self.canvas.coords(self.selection, start_x_new, start_y_new, end_x_new, end_y_new)


imageCompleter = ImageInpaint()
