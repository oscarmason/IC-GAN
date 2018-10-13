from network_settings import *
import tensorflow as tf
import os

def load_checkpoint(sess):
    """
    Checks whether a previously learnt model is available to either invoke or continue training from, and if so loads
    that model

    :argument
        sess:   Current TensorFlow session

    :returns
        save:   Saver object used for saving the weights as the network trains

    """

    saver = tf.train.Saver(max_to_keep=1)

    checkpoint = tf.train.get_checkpoint_state(os.path.dirname(model_path))
    print("Loading model...")

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Model restored")
    else:
        print("WARNING: Model not restored")

    return saver