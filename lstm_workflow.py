import data.get_data as data
import data.preprocess as dp
import models.lstm as lstm

# %%

def evaluate(model, X_test, y_test):
    pass


def render_plot(history):
    pass


def main():
    test_df = data.get_test_data()
    train_df = data.get_train_data()

    (X_train, y_train), (X_test, y_test), sequence_length = dp.preprocess_data(train_df, test_df)

    model, history = lstm.build_classifier(500, X_train, y_train, X_test, y_test)
    evaluate(model, X_test, y_test)
    render_plot(history)


if __name__ == '__main__':
    main()
