import sys

import tensorflow.keras
import tensorflow as tf


def tensorflow_info_version():
    """
    Print version of Tensorflow and Keras
    """

    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {tensorflow.keras.__version__}")
    print()
    print(f"Python {sys.version}")


def gpu_is_available():
    """
    Returns True if Tensorflow is installed and uses GPU
    """
    return tf.test.is_gpu_available()
