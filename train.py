from network_settings import *
from image_processing import ImageProcessor
from network import Network

import glob
import tensorflow as tf
import numpy as np
import model_import as mi


class Train:
    """
    Train

    Train is responsible for carrying out the training process. This includes:
        - Loading of the dataset
        - Calling methods to carry out any necessary pre-processing steps such as masking the training images
        - Saving and restoring the learnt models
        - Running the specified number of epochs to optimise both the generator and discriminator

    """

    image_processor = ImageProcessor()
    training_dataset_path = root + './datasets/training/*'

    def train(self):
        """
        Trains the network on the requested dataset for a set number of epochs

        """

        # Retrieve the tensors from the network
        network = Network()
        d_input, g_input, g_output, g_output_patch_only, d_optimizer, g_optimizer, surrounding_region, \
            patch_ground_truth, d_cost_fake, d_cost_real, g_cost, training = network.network(batch_size)

        # Create a new TensorFlow session
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = mi.load_checkpoint(sess)

        # Get the paths of all the files within the training dataset
        file_paths = np.array(glob.glob(self.training_dataset_path))
        number_of_instances = len(file_paths)

        for epoch in range(number_of_epochs):

            # Shuffle images
            file_paths = file_paths[np.random.permutation(number_of_instances)]

            # Iterate through each batch of images
            for i in range(number_of_instances // batch_size):

                # Retrieve batch of training images_____________________________________________________________________

                batch_file_paths = file_paths[i * batch_size : i * batch_size + batch_size]
                d_batch, g_batch, full_image_batch, surrounding_region_batch, patch_ground_truth_batch = \
                    self.image_processor.create_batch(batch_file_paths)


                # Optimise discriminator and generator__________________________________________________________________
                _ = sess.run([d_optimizer], feed_dict={g_input: g_batch, surrounding_region: surrounding_region_batch,
                                       d_input: d_batch, training: True})
                _ = sess.run([g_optimizer], feed_dict={g_input: g_batch, surrounding_region: surrounding_region_batch,
                                       d_input: d_batch, patch_ground_truth: patch_ground_truth_batch, training: True})


                # Calculate and print error_____________________________________________________________________________
                if i % 10 == 0:
                    d_error_real = d_cost_real.eval({d_input: d_batch, training: True})

                    d_error_fake = d_cost_fake.eval({g_input: g_batch, surrounding_region: surrounding_region_batch,
                                                    patch_ground_truth: patch_ground_truth_batch, training: True})

                    g_error = g_cost.eval({g_input: g_batch, surrounding_region: surrounding_region_batch,
                                                    patch_ground_truth: patch_ground_truth_batch, training: True})

                    print(epoch, i, d_error_real, d_error_fake, g_error)


                # Save model____________________________________________________________________________________________

                if i % 1000 == 0:
                    saver.save(sess, model_path + '/model.ckpt')


train = Train()
train.train()