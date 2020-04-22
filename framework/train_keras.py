import sys
import warnings
import time
import os.path
warnings.filterwarnings("ignore")
sys.path.insert(1, 'D:\\res2020\Computer_Vision\\violence_detect\\root')
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models.keras_models import ResearchModels
class train_using_keras:
    def train(seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=10,):
    
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # # Get the data and process it.
    # if image_shape is None:
    #     data = DataSet(
    #         seq_length=seq_length,
    #         class_limit=class_limit
    #     )
    # else:
    #     data = DataSet(
    #         seq_length=seq_length,
    #         class_limit=class_limit,
    #         image_shape=image_shape
    #     )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = 64

    # if load_to_memory:
    #     # Get data.
    #     X, y = data.get_all_sequences_in_memory('train', data_type)
    #     X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    # else:
    #     # Get generators.
    #     generator = data.frame_generator(batch_size, 'train', data_type)
    #     val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Fit!

     rm.model.fit(
        X,
        Y,
        batch_size=batch_size,
        validation_data=(X_eval, Y_eval),
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger],
        epochs=nb_epoch)



