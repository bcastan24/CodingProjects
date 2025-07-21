import keras as keras
from keras import layers

def buildAndCompileModel(xTrain):
    model = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(xTrain.shape[1],)), layers.Dense(64, activation='relu'), layers.Dense(64, activation='relu'),layers.Dense(1)])
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001), metrics=['mae'])
    return model

def trainNN(nnModel, xTrain, yTrain, xTest, yTest):
    print("\nStarting Training...")
    history = nnModel.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=1000, batch_size=30, verbose=1)
    return nnModel

def makePredictions(nnModel, xTest, yTest):
    predictions = nnModel.predict(xTest)
    avgDiff = 0.0
    print(f"\nSample predictions vs Actual")
    for j in range(50):
        print(
            f"Predicted: {predictions[j][0] + 2.0:.3f}, Actual: {yTest[j] + 2.0:.3f}, Difference: {(predictions[j][0] + 2.0) - (yTest[j] + 2.0):.3f}")
        avgDiff += (predictions[j][0] + 2.0) - (yTest[j] + 2.0)
    print(f"Average Difference: {abs(avgDiff / 50):.3f}")