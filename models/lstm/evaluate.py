import matplotlib.pyplot as plt


def evaluate(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print("Accuracy: %.2f" % (acc))


def render_plot(model_name, history):
    plt.figure(figsize=(4, 5))
    plt.plot(history['accuracy'], label='Train accuracy')
    plt.plot(history['loss'], label='Train loss')

    plt.title(model_name, fontsize=14)
    plt.legend()
    plt.savefig(f"plots/{model_name}.png")
    plt.show()
