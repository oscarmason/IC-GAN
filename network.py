from network_settings import *
from layers import conv_transpose
from layers import convolution
import numpy as np
import tensorflow as tf



class Network:

    """
    The Network class holds the neural network used to train the model. It consists of both the generator and
    discriminator, the optimisation methods used for training, and helper functions required to insert the generated
    patch back into the original image

    """

    def discriminator(self, input, training, reuse):
        """
        The discriminator is responsible for taking an image and predicting whether or not it contains a generated patch

        Both the narrow and wide path take in the same input image, and merge together before carrying out a couple
        more convolutions and making a final decision

        :argument
            input:      The input could either be a real image or one containing a patch that was generated
            training:   Whether the network is training or not
            reuse:      Whether the weights in all the layers are to be reused

        :returns
            output:     Prediction on whether or not the discriminator believes an image contains a generated patch or
                        not. If output is 0, then the network is 100% certain the image is real. If it is 1, then it is
                        100% certain it contains a patch
        """

        with tf.variable_scope("discriminator", reuse=reuse):

            # Narrow Path_______________________________________________________________________________________________

            layer = convolution(input, 128, (5, 5), (1, 1), 1, training, reuse, dilation_rate=(3,3))

            layer = convolution(layer, 128, (5, 5), (2, 2), 2, training, reuse)

            layer = convolution(layer, 128, (5, 5), (1, 1), 3, training, reuse, dilation_rate=(3,3))

            layer = convolution(layer, 128, (5, 5), (1, 1), 4, training, reuse, dilation_rate=(4,4))

            layer = convolution(layer, 128, (5, 5), (2, 2), 5, training, reuse)

            layer = convolution(layer, 128, (5, 5), (2, 2), 6, training, reuse)

            layer = convolution(layer, 128, (5, 5), (1, 1), 7, training, reuse, dilation_rate=(4,4), batch_norm=True)


            # Wide Path_________________________________________________________________________________________________

            layer_wide = convolution(input, 128, (10, 10), (1, 1), 8, training, reuse)

            layer_wide = convolution(layer_wide, 128, (10, 10), (1, 1), 9, training, reuse)

            layer_wide = convolution(layer_wide, 128, (9, 9), (2, 2), 10, training, reuse)

            layer_wide = convolution(layer_wide, 128, (9, 9), (1, 1), 11, training, reuse, padding='VALID')

            layer_wide = convolution(layer_wide, 128, (5, 5), (1, 1), 12, training, reuse)

            layer_wide = convolution(layer_wide, 128, (5, 5), (1, 1), 13, training, reuse, padding='VALID',
                                     batch_norm=True)

            #___________________________________________________________________________________________________________

            layer = tf.concat([layer, layer_wide], 3)

            layer = convolution(layer, 128, (5, 5), (1, 1), 14, training, reuse)

            layer = convolution(layer, 128, (5, 5), (1, 1), 15, training, reuse, batch_norm=True)

            layer = tf.reshape(layer, (-1, 4 * 4 * 128))

            output = tf.layers.dense(layer, 1, name='d_16')

            return output


    def generator(self, input, surrounding_region, sparse, training, reuse=False):
        """
        The generator is responsible for taking a masked image and producing a plausible patch. Both the narrow and wide
        paths take the same image as their input, while the shallow layer takes segments from the top, bottom, left and
        right regions which surround the masked selection.

        :argument
            input:                  Image input containing a masked out region
            surrounding_region:     Pixels surrounding the masked out region, cropped to the input size of the
                                    discriminator
            sparse:                 A sparse tensor used to extract the generated patch from the generator output and
                                    merge it into the surrounding pixels of the original image
            training:               Whether the network is currently training or not
            reuse:                  Whether the learnt weights should be reused or not

        :returns
            image:                  Contains the patch along with the surrounding pixels in the original image cropped
                                    to the input size of the discriminator
            patch:                  The generated patch

        """
        with tf.variable_scope("generator", reuse=reuse):

            # Shallow Path______________________________________________________________________________________________
            #
            # The shallow path consists of two convolution layers.
            # The first layer takes segments from above and below the image which are then concatenated along the Y-axis
            # The second take segments from the left and right of the image which are concatenated along the X-axis
            # Following the process of a single convolution, these layers are concatenated with the output from the
            # narrow and wide path a couple of layers prior to the output layer

            y_axis = 1
            x_axis = 2

            image_top = tf.slice(input, [0, 0, 8, 0], [-1, 16, 32, channels])
            image_bottom = tf.slice(input, [0, 32, 8, 0], [-1, 16, 32, channels])
            layer_shallow = tf.concat([image_top, image_bottom], y_axis)

            layer_shallow = convolution(layer_shallow, 128, (3, 3), (1, 1), 2, training, reuse, dilation_rate=(3,3),
                                        batch_norm=True)

            image_left = tf.slice(input, [0, 8, 0, 0], [-1, 32, 16, channels])
            image_right = tf.slice(input, [0, 8, 32, 0], [-1, 32, 16, channels])
            layer_shallow_ = tf.concat([image_left, image_right], x_axis)

            layer_shallow_ = convolution(layer_shallow_, 128, (3, 3), (1, 1), 1, training, reuse, dilation_rate=(3,3),
                                         batch_norm=True)


            # Narrow Path_______________________________________________________________________________________________

            layer = convolution(input, 128, (5, 5), (1, 1), 3, training, reuse, dilation_rate=(3,3))

            layer = convolution(layer, 128, (5, 5), (1, 1), 4, training, reuse, dilation_rate=(3,3))

            layer = convolution(layer, 128, (3, 3), (2, 2), 5, training, reuse)

            layer = convolution(layer, 128, (3, 3), (1, 1), 6, training, reuse, dilation_rate=(3,3))

            layer = conv_transpose(layer, 128, (5, 5), (1, 1), 7, training, padding="VALID")

            layer = convolution(layer, 128, (3, 3), (1, 1), 8, training, reuse, dilation_rate=(2,2))

            layer = convolution(layer, 128, (3, 3), (1, 1), 9, training, reuse, dilation_rate=(4,4))

            layer = convolution(layer, 128, (3, 3), (1, 1), 10, training, reuse, dilation_rate=(8,8))

            layer = conv_transpose(layer, 128, (5, 5), (1, 1), 11, training, padding="VALID")

            layer = convolution(layer, 128, (5, 5), (1, 1), 12, training, reuse, dilation_rate=(4,4), batch_norm=True)


            # Wide Path_________________________________________________________________________________________________

            layer_wide = convolution(input, 128, (8, 8), (1, 1), 13, training, reuse)

            layer_wide = convolution(layer_wide, 128, (8, 8), (1, 1), 14, training, reuse)

            layer_wide = convolution(layer_wide, 128, (9, 9), (1, 1), 15, training, reuse, padding='VALID')

            layer_wide = convolution(layer_wide, 128, (8, 8), (1, 1), 16, training, reuse, dilation_rate=(6,6))

            layer_wide = convolution(layer_wide, 128, (9, 9), (1, 1), 17, training, reuse, padding='VALID')

            layer_wide = convolution(layer_wide, 128, (8, 8), (1, 1), 18, training, reuse, dilation_rate=(6,6))

            layer_wide = convolution(layer_wide, 128, (8, 8), (1, 1), 19, training, reuse, batch_norm=True)

            #___________________________________________________________________________________________________________


            layer = tf.concat([layer, layer_wide, layer_shallow, layer_shallow_], 3)

            layer = convolution(layer, 128, (3, 3), (1, 1), 20, training, reuse)

            layer = convolution(layer, 64, (3, 3), (1, 1), 21, training, reuse, batch_norm=True)

            layer = convolution(layer, channels, (5, 5), (1, 1), 22, training, reuse, activation=False)

            layer = tf.nn.sigmoid(layer)

            patch = layer * 255
            image = self.merge_original_image_with_generated(surrounding_region, patch, sparse)

            patch = tf.slice(image, [0, D_patch_margin_size, D_patch_margin_size, 0],
                             [-1, patch_width, patch_width, channels])

        return image, patch


    def network(self, batch_size=1):
        # d_input, g_input, g_output, g_output_patch_only, d_optimizer, g_optimizer, surrounding_region, \
        #     patch_ground_truth, d_cost_fake, d_cost_real, d_cost, g_cost_gan, g_cost_mse, g_cost, training

        """
        Network is responsible for calculating and minimising the cost of both the generator and discriminator

        :argument
            batch_size:         Number of images in the current batch

        :returns
            d_input:                Input to the discriminator
            g_input:                Input to the generator
            g_output:               Generated patch merged back into the surrounding image
            g_output_patch_only:    Generated patch
            d_optimizer:            Optimiser for training the discriminator
            g_optimizer:            Optimiser for training the generator
            surrounding_region:     Area surrounding the masked region, cropped to the size of discriminator's input
            training:               Whether or not the network is currently in train mode

        """

        # Tensors_________________________________________________________________________________________________
        #
        # Values must be fed into these tensors at training or invocation time as follows
        # d_input:              Real or generated image for the discriminator
        # g_input:              Image containing a masked region to be completed
        # surrounding_region:   Pixels surrounding the masked region, cropped to the size of the input to the
        #                       discriminator
        # patch_ground_truth:   Ground truth for the masked region. This is compared against during training and the
        #                       loss minimised

        d_input = tf.placeholder(tf.float32, [None, D_input_size, D_input_size, channels])
        g_input = tf.placeholder(tf.float32, [None, G_input_size, G_input_size, channels])
        surrounding_region = tf.placeholder(tf.float32, [None, D_input_size, D_input_size, channels])
        patch_ground_truth = tf.placeholder(tf.float32, [None, patch_width, patch_width, channels])
        training = tf.placeholder(tf.bool, [])


        # Generator and Discriminator___________________________________________________________________________________

        g_output, g_output_patch_only = self.generator(g_input, surrounding_region,
                                              self.setup_sparse(batch_size=batch_size), training=training, reuse=False)

        d_output_real = self.discriminator(d_input, training=training, reuse=False)

        d_output_fake = self.discriminator(g_output, training=training, reuse=True)


        # Optimise Discriminator________________________________________________________________________________________

        d_cost_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_output_real, labels=tf.zeros_like(d_output_real)))

        d_cost_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_output_fake, labels=tf.ones_like(d_output_fake)))

        d_cost = d_cost_real + d_cost_fake


        # Optimise Generator____________________________________________________________________________________________

        g_cost_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_output_fake, labels=tf.zeros_like(d_output_fake)))

        g_cost_mse = tf.reduce_mean(tf.losses.mean_squared_error(patch_ground_truth, g_output_patch_only))

        g_cost = g_cost_gan + g_cost_mse


        # Set training variables________________________________________________________________________________________
        #
        # During training, the generator and discriminator are trained in turn. Therefore, the only weights that should
        # be modified are those belonging to the one currently being trained. By passing the filtered list of weights,
        # d_variables and g_variables, to the optimizers this constraint can be applied
        training_variables = tf.trainable_variables()

        d_variables = [var for var in training_variables if 'discriminator' in var.name]
        g_variables = [var for var in training_variables if 'generator' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            d_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(d_cost, var_list=d_variables)
            g_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(g_cost, var_list=g_variables)

        a = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        return d_input, g_input, g_output, g_output_patch_only, d_optimizer, g_optimizer, surrounding_region, \
            patch_ground_truth, d_cost_fake, d_cost_real, g_cost, training


    def setup_sparse(self, batch_size):
        """
        Setup sparse creates a tensor with values at the specific locations specified.
        This is used to merge the generated patch with the original image before being fed into the discriminator.
        All pixels with a value of 1 will be kept, and all those with zero will be replaced by the surrounding pixels in
        the original image. The final sparse will be a square containing zeros around the border with a smaller square
        of ones in the centre of side length equal to the patch width

        :argument
            batch_size:         Number of images in each training batch

        :returns
            sparse:             A tensor consisting of all zeros, apart from a square in the centre, the same size as
                                the patch width, which are set to the value of one

        """
        location = []
        for n in range (batch_size):
            for i in range (D_patch_margin_size, D_patch_margin_size + patch_width):
                for j in range (D_patch_margin_size, D_patch_margin_size + patch_width):
                    for k in range (channels):
                        location.append([n, i, j, k])

        values = np.ones((batch_size * patch_width * patch_width * channels))
        shape = [batch_size, D_input_size, D_input_size, channels]
        sparse = tf.SparseTensor(location, values, shape)

        return sparse


    def merge_original_image_with_generated(self, masked_image, generated_image, sparse):
        """
        Inserts the generated patch into the surrounding pixels of the original image

        :argument
            masked_image:           Original masked image in which the generated patch is to be inserted
            generated_image:        Cropped image containing the generated patch
            sparse:                 Sparse tensor used to allow quick merging of both images via simple addition
        :returns
                                    Image containing the generated patch

        """
        return tf.cast((tf.sparse_tensor_to_dense(sparse)), tf.float32) * generated_image + masked_image
