from models.lstm import lstm_predict
from models.lstm import lstm_model
import data.get_data as data
import data.preprocess as preprocess


class Classifier:
    """
    Simple class wraps classifier predicting category on random text
    """
    def __init__(self, basedir="./"):
        self.model = lstm_model.load_local_model("LSTM_DROPOUT", basedir=basedir)
        self.train_df = data.get_train_data(basedir=basedir)
        test_df = data.get_test_data(basedir=basedir)
        self.tokenizer = preprocess.common_tokenizer(self.train_df, test_df)

    def get_category(self, question):
        return lstm_predict.predict_lstm(self.model, self.tokenizer, self.train_df, question)


if __name__ == '__main__':
    classifier = Classifier()
    print(classifier.get_category("What is the oldest profession ?"))
