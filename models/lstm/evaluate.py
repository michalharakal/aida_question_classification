import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


def evaluate_lstm(model, X_test, y_test, df_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print("Accuracy: %.2f" % acc)

    y_pred = model.predict(X_test, batch_size=64, verbose=1)
    y_pred_class_index = np.argmax(y_pred, axis=1)
    test_categories = pd.get_dummies(df_test["category"]).columns
    predicted_class_label = np.array([test_categories[index] for index in y_pred_class_index])
    report = classification_report(df_test["category"], predicted_class_label, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(2)
    # save df results to cvs for later report
    report_df.to_csv('./report/' + model.name + '.csv')
    return report_df




def render_plot(model_name, history):
    plt.figure(figsize=(4, 5))
    plt.plot(history['accuracy'], label='Train accuracy')
    plt.plot(history['loss'], label='Train loss')

    plt.title(model_name, fontsize=14)
    plt.legend()
    plt.savefig(f"plots/{model_name}.png")
    plt.show()
