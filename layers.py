"""
This Module contains the following layers which are used by the network:
    - Convolution
    - Convolution Transpose

"""
import tensorflow as tf


def convolution(input, filters, kernel_size, strides, layer_id, training, reuse, dilation_rate=(1, 1),
                padding='SAME', activation=True, batch_norm=False):
    """
    Creates a convolutional layer

    :argument
        input:          Input to which the kernel is applied
        filters:        Number of output channels to this layer
        kernel_size:    Size of the kernel. Can be an integer to specify the same size for all dimensions or a tuple
                        to specify individual sizes for each dimension
        strides:        The number of strides the kernel should take on each move
        layer_id:       A unique id which identifies the layer within the network
        training:       Whether or not the network is currently training. This is required for batch normalisation since
                        it works differently during training than when being invoked
        reuse:          Whether the weights should be reused. In this case of this particular network, the weights in
                        the discriminator are reused between the two instances required for both the real and generated
                        images
        dilation_rate:  Dilation rate to be applied to the convolution. The default value, (1,1) is normal convolution
                        in which no padding is inserted between the weights
        padding:        Whether to pad the input around the edge or not. 'SAME' padding should be used if the output is
                        to remain the same size as the input
        activation:     When activation is True, ReLU will be applied before returning the output
        batch_norm:     When batch_norm is True, batch normalisation will be applied following convolution

    :returns
        layer:          The output tensor from this layer

    """
    layer = tf.layers.conv2d(input, filters, kernel_size, strides, padding=padding, activation=None,
                             dilation_rate=dilation_rate, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name=str(layer_id), reuse=reuse)

    if batch_norm:
        layer = batch_normalisation(layer, training, name="_batch_norm" + str(layer_id))

    if activation:
        layer = tf.nn.relu(layer)

    return layer


def conv_transpose(input, filters, kernel_size, strides, layer_id, training, padding="VALID"):
    """
    Creates a convolutional transpose layer

    :argument
    input:          Input to which the kernel is applied
    filters:        Number of output channels to this layer
    kernel_size:    Size of the kernel. Can be an integer to specify the same size for all dimensions or a tuple
                    to specify individual sizes for each dimension
    strides:        The number of strides the kernel should take on each move
    layer_id:       A unique id which identifies the layer within the network
    training:       Whether or not the network is currently training. This is required for batch normalisation since
                    it works differently during training than when being invoked
    padding:        Whether to pad the input around the edge or not. 'SAME' padding should be used if the output is
                    to remain the same size as the input

    :returns
        layer:       The output tensor from this layer

    """

    layer = tf.layers.conv2d_transpose(input, filters, kernel_size, strides, padding=padding, name=str(layer_id),
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.constant_initializer())

    layer = batch_normalisation(layer, training, name='batch_norm' + str(layer_id))
    layer = tf.nn.relu(layer)
    return layer


def batch_normalisation(input, training, name):
    """
    Carries out batch normalisation on the input

    :argument
        input:      Input to which batch normalisation is applied
        training:   Whether or not the network is currently training. This is required for batch normalisation since
                    it works differently during training than when being invoked
        name:       Unique name which identifies this batch normalisation call

    :returns
                    The output of applying batch normalisation

    """
    return tf.layers.batch_normalization(input, training=training, name=name)