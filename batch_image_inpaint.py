from image_processing import ImageProcessor
from network import Network
from network_settings import *

import glob
import tensorflow as tf
import numpy as np
import model_import as mi


class BatchImageInpaint:
    """
    Batch Image Inpaint is used to invoke the learnt model to complete a batch of images whose missing region is located
    in the same place. This was used for evaluating the quality of the patches produced on a large number of images
    """

    test_dataset_location = root + 'datasets/test/*'

    def generatePatch(self):
        """
        Completes a batch of masked out images

        """

        image_processor = ImageProcessor()

        # Load the network______________________________________________________________________________________________
        #     - g_input: Input to the generator
        #     - g_output_patch_only: Patch generated
        #     - surrounding_region: Region surrounding the masked image to be merged with the generated patch
        #     - training: Whether the model is training or not. When invoking the model, False should be passed in

        network = Network()
        d_input, g_input, g_output, g_output_patch_only, d_optimizer, g_optimizer, surrounding_region, \
            patch_ground_truth, d_cost_fake, d_cost_real, g_cost, training = network.network(batch_size)


        # Create a new TensorFlow session
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())


        # Get the paths of all the files within the test dataset location and shuffle the images
        file_paths = np.array(glob.glob(self.test_dataset_location))
        number_of_instances = len(file_paths)
        indexes = np.random.permutation(number_of_instances)
        file_paths = file_paths[indexes]


        # Load learnt model
        mi.load_checkpoint(sess)


        # Iterate through each batch of images
        for i in range(number_of_instances // batch_size):

            # Retrieve batch of training images
            batch_file_paths = file_paths[i * batch_size: i * batch_size + batch_size]
            _, g_batch, image_full, surrounding_region_batch, _ = image_processor.create_batch(batch_file_paths)

            # Generate patches for the batch of images
            generated_patches = sess.run(g_output_patch_only, feed_dict={g_input: g_batch,
                                         surrounding_region: surrounding_region_batch, training: False})

            # Save the completed images. Both the ground truth (1) and images with the generated patch using unsharp
            # intensities of the default 2.5 and 0.4 are saved
            for k in range(0, batch_size):
                img_id = batch_size * i + k

                image_processor.save_image(image_full[k], img_id, 1)

                generated_patch = generated_patches[k]

                sharpened_patch = image_processor.unsharp_mask(generated_patch)
                sharpened_image = image_processor.merge_patch_with_image(sharpened_patch, image_full[k],
                                                                         patch_startX, patch_startY)
                image_processor.save_image(sharpened_image, img_id, 2)

                sharpened_patch = image_processor.unsharp_mask(generated_patch, 0.5)
                sharpened_image = image_processor.merge_patch_with_image(sharpened_patch, image_full[k],
                                                                         patch_startX, patch_startY)
                image_processor.save_image(sharpened_image, img_id, 3)

            print(i * batch_size)


batch_image_inpaint = BatchImageInpaint()
batch_image_inpaint.generatePatch()
