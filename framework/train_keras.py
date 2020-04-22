import sys
import warnings
import time
import os.path
warnings.filterwarnings("ignore")
sys.path.insert(1, 'D:\\res2020\Computer_Vision\\violence_detect\\root')
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models.keras_models import ResearchModels
class Train_using_keras:
    def train(self,X,Y,X_eval,Y_eval,seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=10):

    
        # Helper: Save the model.
        checkpointer = ModelCheckpoint(
            filepath=os.path.join('data', 'checkpoints', model + '-' + 'images' + \
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
        steps_per_epoch = 64

        # Get the model.
        rm = ResearchModels(2, model, seq_length, saved_model)

        # Fit!

        rm.model.fit(
            X,
            Y,
            batch_size=batch_size,  
            validation_data=(X_eval, Y_eval),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger],
            epochs=nb_epoch)








