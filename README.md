# Inertial Sensor Data Prediction
### Data
This dataset is composed of sensor readings captured from an inertial measurement unit that records both gyroscope and accelerometer data along the x, y, and z axes. The structure is as follows:
1. An initial, unnamed column representing the time steps at which the data was recorded.
2. Three columns containing gyroscope measurements along the x, y, and z axes.
3. Three columns containing acceleration data along the x, y, and z axes.


The data is organized into four primary activity categories: jogging, walking, upstair, and downstair. Additionally, there is a special "falldown" category, which aggregates sensor readings from the four main activities during fall events.
### Training
#### Configuration
* window_size:
This parameter defines the number of time steps included in each data window. A window of 100 time steps is used as an input segment for the model, capturing the sequential behavior of the sensor readings.

* predict_size:
This indicates how many future time steps the model should predict based on the input window. For example, a value of 2 means the model will forecast the next 2 time steps.

* epochs:
The number of complete passes through the training dataset.
This defines the number of samples processed before the model's internal parameters are updated.

* model_type:
Specifies the type of model to be used. The options include:

  * "cnn": Use a Convolutional Neural Network.
  * "lstm": Use a Long Short Term Memory model.

* criterion:
This sets the loss function for training the model.
  * "mse" stands for Mean Squared Error, which measures the average squared difference between predicted and actual values.
  *  "mae" is the Mean Absolute Error (MAE), implemented as L1Loss. This loss function measures the average absolute differences between predicted and actual values. (Less sensitive to outliers comare to mse)

* learning_rate:
The learning rate is a hyperparameter that controls how much to change the model's weights with respect to the loss gradient during training. A value of 0.001 is typically a good starting point.

* early_stopping_patience:
This parameter determines the number of epochs with no improvement in the validation loss after which training will be stopped early. A patience of 10 means that if the model doesn’t improve for 10 consecutive epochs, the training process will halt.

* model_save_path:
This specifies the file path where the best-performing model (according to the validation metrics) will be saved, allowing you to load and use the model later without retraining.
#### Command for running the code
```
python main.py --config <config_name>
```
