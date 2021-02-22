import tensorflow as tf

import optuna
from optuna.integration import TFKerasPruningCallback
import data.get_data as data
import data.preprocess as dp

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import plot_model

BATCHSIZE = 128
CLASSES = 10
EPOCHS = 20
N_TRAIN_EXAMPLES = 3000
STEPS_PER_EPOCH = int(N_TRAIN_EXAMPLES / BATCHSIZE / 10)
VALIDATION_STEPS = 30


def prepare_data(data_column="question", classes_column="category"):
    # get data
    test_df = data.get_test_data()
    train_df = data.get_train_data()
    # preprocess data
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size, tokenizer = dp.preprocess_data(train_df, test_df,
                                                                                                      data_column,
                                                                                                      classes_column)
    return (X_train, y_train), (X_test, y_test), sequence_length, vocab_size, tokenizer


def create_model(trial, seq_length, categories_count, vocab_size):
    # Hyperparameters to be tuned by Optuna.
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.0, 1.0)
    units = trial.suggest_categorical("units", [32, 64, 128, 256, 512])

    model = tf.keras.Sequential(name="model_name")
    model.add(Embedding(input_dim=vocab_size, output_dim=256, input_length=seq_length))
    model.add(LSTM(units=units))
    model.add(Dropout(0.5))
    model.add(Dense(categories_count, activation=tf.nn.softmax))

    # Compile model.
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def objective(trial):
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    # Metrics to be monitored by Optuna.
    if tf.__version__ >= "2":
        monitor = "val_accuracy"
    else:
        monitor = "val_acc"

        # Create dataset instance.
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size, tokenizer = prepare_data()

    # Create tf.keras model instance.
    model = create_model(trial, sequence_length, y_train.shape[1], vocab_size)

   # ds_train = train_dataset()
    # ds_eval = eval_dataset()

    # Create callbacks for early stopping and pruning.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3),
        TFKerasPruningCallback(trial, monitor),
    ]

    # Train model.
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=(X_test, y_test),
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks,
    )

    return history.history[monitor][-1]


def show_result(study):
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def main():

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
    )

    study.optimize(objective, n_trials=25, timeout=600)

    show_result(study)


if __name__ == '__main__':
    main()
