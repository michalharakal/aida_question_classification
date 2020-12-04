import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import os
import glob


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


def training_accuracy_plot(model_name, history):
    """
    Plot and export into the image the accuracy and loss values from history data
    @param model_name: name of the model used for plot title and image file name
    @param history:
    @return:
    """
    plt.figure(figsize=(4, 5))
    plt.plot(history['accuracy'], label='Train accuracy')
    plt.plot(history['loss'], label='Train loss')

    plt.title(model_name, fontsize=14)
    plt.legend()
    plt.savefig(f"plots/{model_name}.png")
    plt.show()


def lstm_report_plot(plot_name='lstm_f1_results.png'):
    files = glob.glob('report/LSTM*.csv')

    df = pd.concat([pd.read_csv(fp).assign(filename=os.path.basename(fp).split(',')[0]) for fp in files])
    df.columns = ['Category', 'precision', 'recall', 'f1', 'support', 'filename']
    df_pv = df.pivot_table(columns=['filename'], index=['Category']).round(2)

    df_f1 = df_pv['f1']
    df_f1.columns = [x.split(".")[0] for x in df_f1.columns]

    df_f1_transpose = df_f1.T

    # %%

    df_f1_transpose.plot()
    plt.title('F1 Scores over different models')
    plt.ylabel('values')
    plt.xticks(rotation=15)
    plt.grid('on')

    plt.savefig(f'./plots/{plot_name}')
    plt.show()
