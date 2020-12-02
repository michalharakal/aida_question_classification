import matplotlib.pyplot as plt


def evaluate(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print("Acurácia: %.2f" % (acc))

    """
        print pipeline results for given pipeline and X, y values

        :param pipeline: fitted sklearn pipeline
        :param X_test: X_test values
        :param y_test: y_test values
        :return: None
        """

    pass


def render_plot(history):
    plt.figure(figsize=(4, 5))
    plt.plot(history.history['accuracy'], label='Train accuracy')
    plt.plot(history.history['loss'], label='Train loss')

    plt.title("Title", fontsize=14)
    plt.legend()
    plt.savefig("plots/lstm1.png")
    plt.show()
