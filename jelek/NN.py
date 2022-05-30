import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from tensorflow import keras
from tensorflow.keras import callbacks,layers,utils
import keras_tuner as kt
import visualkeras
import matplotlib.pyplot as plt
from plot_keras_history import plot_history

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten())

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def buil_model():
    model = keras.Sequential(name="Emotions_model")
    model.add(layers.Dense(2033, input_shape=[4], activation='softmax'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(1033, activation='softmax'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(33, activation='softmax'))
    #model.add(layers.BatchNormalization())

    model.add(layers.Dense(3, activation='softmax'))

    return model

def pre_process_data():
    df = pd.read_csv('C:/Users/Asus/Desktop/allamviszga/programok/jelek/my_data.csv')

    # megnezem , hogy hany darab adatom van mindegyikbol
    # print(df['label'].value_counts())

    # szetvalasztom az adatokat a label-tol
    dataframe = df.drop('label', axis=1)
    y = df['label']

    # mennyire kulonboznek az adatok, nincs fuggoseg kozottuk
    mutual_info = mutual_info_classif(dataframe, y)
    # print(f"mutual_info : {mutual_info}")

    # 0,1 alakitom label szerint
    y = pd.get_dummies(df['label'])
    #print(f"y : {y}")

    # pandas sorozatta alakitom
    mutual_info = pd.Series(data=mutual_info, index=dataframe.columns)
    mutual_info = (mutual_info * 100).sort_values(ascending=False)
    # print(f"mutual_info : {mutual_info}")

    # skalazom az adatokat es transzformacio ala vetem
    scaled_data = StandardScaler().fit_transform(dataframe[:])
   #print(f"scaled_data:\n {scaled_data}")

    X_train, x_test, Y_train, y_test = train_test_split(scaled_data, y, random_state=43, test_size=0.3)
    # print(f"X_train:\n {X_train} \n Y_train: \n {Y_train}")
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, random_state=43, test_size=0.3)

    return x_train, x_val, y_train, y_val,x_test, y_test

if __name__=='__main__':
    x_train, x_val, y_train, y_val,x_test, y_test = pre_process_data()
    # model = buil_model()
    #
    # rms = keras.optimizers.RMSprop(learning_rate = 0.00085)
    # adamax=keras.optimizers.Adamax(learning_rate = 0.00085, beta_1 = 0.9, beta_2 = 0.999)
    #
    # model.compile(optimizer=rms,loss='categorical_crossentropy',metrics=['accuracy'])
    #
    # #visualkeras.layered_view(model, legend= True, to_file='output.png').show()

    tuner = kt.Hyperband(build_model,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')
    call=callbacks.EarlyStopping(patience=10,min_delta=0.0001,restore_best_weights=True)
    tuner.search(x_train, y_train.to_numpy(), epochs=50, validation_split=0.2, callbacks=[call])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x_train, y_train.to_numpy(), epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(x_train, y_train.to_numpy(), epochs=best_epoch, validation_split=0.2)
    eval_result = hypermodel.evaluate(x_test, y_test.to_numpy())
    print("[test loss, test accuracy]:", eval_result)
    #history=model.fit(x_train,y_train.to_numpy(),validation_data=(x_val,y_val),batch_size=20,epochs=30,callbacks=[call])

    #model.evaluate(x_test,y_test.to_numpy())

    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])  # RAISE ERROR
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])  # RAISE ERROR
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()


