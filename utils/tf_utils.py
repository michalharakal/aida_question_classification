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


def activate_compatibility_allow_growth():
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


if __name__ == '__main__':
    gpu_is_available()
