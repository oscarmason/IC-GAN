from network_settings import *
from scipy import misc

import cv2
import numpy as np


class ImageProcessor:
    """
    Image Processor contains methods used for:
        - Importing and preparing the datasets
        - Merging the generated patch into the rest of the image
        - Applying the unsharp mask algorithm
        - Saving the completed images

    """
    def create_batch(self, paths):
        """
        Creates all the necessary images for a single batch on which the network trains on, as well as images used to
        merge the generated with the original

        :argument
            paths                     Paths of the images which form the current batch being created

        :returns
            d_batch:                  Contains the original images cropped to include a small region around the selected
                                      patch. This is used by the discriminator to learn what a real image looks like
            g_batch:                  Contains cropped images with a masked out region. Used by the generator to learn
                                      to complete missing regions
            d_full_image_batch:       Contains the full original images with a mask applied. The generated patches are
                                      merged back into these image to form the completed image to present to the user
            surrounding_region_batch: Contains the original images with a mask applied, cropped to the input size of the
                                      discriminator. These are merged with the output of the generator before being fed
                                      into the discriminator
            patch_ground_truth_batch: The original patch which was masked over. Used to carry out MSE on the generated
                                      patch

        """
        images = self.load_dataset(paths)

        g_batch = np.asarray(images)

        # Extract patch from original image for the generator to train on
        g_batch, patch_ground_truth_batch = self.extract_patch_batch(g_batch, patch_startX, patch_startY)

        # Exstract a cropped area that surround the masked out region
        surrounding_region_batch = np.array(g_batch)
        surrounding_region_batch = self.crop_images(surrounding_region_batch, D_patch_margin_size,
                                                    patch_startX, patch_startY)

        # Remove the surrounding region of the images
        d_batch = np.array(images)
        d_batch = self.crop_images(d_batch, D_patch_margin_size, patch_startX, patch_startY)

        d_full_image_batch = np.array(images)

        g_batch = self.crop_images(g_batch, G_patch_margin_size, patch_startX, patch_startY)

        return d_batch, g_batch, d_full_image_batch, surrounding_region_batch, patch_ground_truth_batch


    def load_dataset(self, paths):
        """
        Loads the images from the specified location
        :param
            paths:      Paths of the images to import

        :return
            images:     Numpy array containing the pixels values of the images requested
        """
        images = []
        for image_path in paths:
            image = misc.imread(image_path)
            images.append(image)

        return images


    def create_image_components(self, img, start_x, start_y):
        """
            Creates the necessary image components required to complete the selected patch
        :param
            img                         Img to create the components for
            start_x:                    Start position of the masked area along the x dimension
            start_y:                    Start position of the masked area along the y dimension
        :return
            g_batch:                    Masked image cropped to the output size of the generator
            masked_image                Full image containing the masked area
            surrounding_region_batch    Masked image cropped to the input size of the discriminator

        """

        # Remove a patch from the image at the requested start positions
        g_batch = np.asarray([img])
        g_batch, _ = self.extract_patch_batch(g_batch, start_x, start_y)

        surrounding_region_batch = np.array(g_batch)
        masked_image = np.array(g_batch)

        # Crop image to the input size of the discriminator
        surrounding_region_batch = self.crop_images(surrounding_region_batch, D_patch_margin_size, start_x, start_y)

        # Crop image to the out size of the generator
        g_batch = self.crop_images(g_batch, G_patch_margin_size, start_x, start_y)

        return g_batch, masked_image, surrounding_region_batch


    def crop_image(self, img, margin_size, start_x, start_y):
        """
        Crops the image around the selected region with a margin size as specified
        :param
            img:            Image to be cropped
            margin_size:    Width of the area to leave in place around the masked region
            start_x:        Start position of the masked region along the x dimension
            start_y:        Start position of the masked region along the y dimension
        :return
            img:            The final cropped image

        """
        img = np.array(img)
        radius_startY = start_y - margin_size
        radius_startX = start_x - margin_size
        radius_endY = radius_startY + margin_size * 2 + patch_width
        radius_endX = radius_startX + margin_size * 2 + patch_width

        img = img[radius_startY : radius_endY, radius_startX : radius_endX, :]

        return img


    def crop_images(self, images, margin_size, start_x, start_y):
        """
        Crops a batch of images around the selected region with a margin size as specified
        :param
            images:         Batch of images to crop
            margin_size:    Width of the area to leave in place around the masked regions
            start_x:        Start position of the masked region along the x dimension
            start_y:        Start position of the masked region along the y dimension
        :return
            cropped_images: Images cropped to the requested size

        """
        cropped_images = np.empty((images.shape[0], margin_size * 2 + patch_width,
                                       margin_size * 2 + patch_width, channels))

        for i in range(images.shape[0]):
            cropped_images[i] = self.crop_image(images[i], margin_size, start_x, start_y)
        return cropped_images


    def extract_patch(self, image, start_x, start_y):
        """
        Removes the selected region from the image and returns both the removed patch and the masked image
        :param
            image:              Image from which to mask a region
            start_x:            Start position of the masked region along the x dimension
            start_y:            Start position of the masked region along the y dimension
        :return
            image:              Image containing a masked out region
            patch_ground_truth: Ground truth of the region that was removed from the images
        """
        patch_ground_truth = np.zeros((patch_width, patch_width, channels))
        patch_i = 0
        patch_j = 0

        for i in range(start_y, start_y + patch_width):

            for j in range (start_x, start_x + patch_width):
                patch_ground_truth[patch_i][patch_j] = image[i][j]
                image[i][j] = 0
                patch_j += 1

            patch_j = 0
            patch_i += 1

        return image, patch_ground_truth


    def extract_patch_batch(self, images, start_x, start_y):
        """
        Removes the selected region from a batch of images and returns both the removed patches and the masked images
        :param
            images:             Images from which to mask a region
            start_x:            Start position of the masked region along the x dimension
            start_y:            Start position of the masked region along the y dimension
        :return
            images:             Images containing a masked out region
            patch_ground_truth: Ground truth of the region that was removed from the images
        """
        number_of_instances = images.shape[0]
        patch_ground_truth = np.zeros((number_of_instances, patch_width, patch_width, channels))

        for i in range(images.shape[0]):
            images[i], patch_ground_truth[i] = self.extract_patch(images[i], start_x, start_y)

        return images, patch_ground_truth


    def merge_patch_with_image(self, generated_patch, full_image, start_x, start_y):
        """
        Inserts the generated patch into the original full image
        :param
            generated_patch:    Generated patch to be inserted into the original image
            full_image:         Full image into which the generated patch is to be inserted
            start_x:            Start position of the masked region along the x dimension
            start_y:            Start position of the masked region along the y dimension
        :return
            merged_image:       Original image containing the generated patch
        """
        full_image = np.squeeze(full_image)
        image_height = full_image.shape[0]
        image_width = full_image.shape[1]

        top = full_image[0:start_y, 0:image_width]
        left = full_image[start_y: start_y + patch_width, 0:start_x]
        right = full_image[start_y: start_y + patch_width, start_x + patch_width:image_width]
        bottom = full_image[start_y + patch_width:image_height, 0:image_width]


        merged_image = np.append(left, generated_patch, 1)
        merged_image = np.append(merged_image, right, 1)
        merged_image = np.append(top, merged_image, 0)
        merged_image = np.append(merged_image, bottom, 0)

        return merged_image


    def unsharp_mask(self, img, strength=2.2):
        """
        :argument
            img:                Image to be sharpened
            strength:           Extent at which to sharpen the image. The higher the value, the more the image is
                                sharpened. A value of 0 has no affect
        :returns
            img:                The sharpened image

        """
        original_img = img
        img_weight = 1 + strength
        gaussian_weight = 0 - strength

        blurred_image = cv2.GaussianBlur(img, (3, 3), 11.0)
        img = cv2.addWeighted(img, img_weight, blurred_image, gaussian_weight, 0)

        img[img < 0] = 0
        img[img > 255] = 255

        img[0,:,:] = original_img[0,:,:]
        img[patch_width-1,:,:] = original_img[patch_width-1,:,:]
        img[:,0,:] = original_img[:,0,:]
        img[:,patch_width-1,:] = original_img[:,patch_width-1,:]

        return img


    def save_image(self, img, img_id, type):
        """
        :param
            img:    Image to be saved
            img_id: ID to differentiate each image
            type:   Allows multiple saves of the same image. Can be used to differentiate between generated, sharpened
                    and ground truth images

        """
        misc.imsave(root + 'samples/{:02d}_{:02d}'.format(img_id, type) + '.png', (img).astype(np.uint8))
