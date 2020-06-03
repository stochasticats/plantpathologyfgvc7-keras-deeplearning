# Predicting foliar disease with Deep Learning in Keras
Deep Learning Model in Keras is used for image classification on the Plant Pathology 2020 - FGVC7 competition data set to identify the category of foliar diseases in apple trees.

## Some images from data set

![alt text](https://github.com/stochasticats/plantpathologyfgvc7-keras-deeplearning/blob/master/train_images.png "Some images from the training set")

![alt text](https://github.com/stochasticats/plantpathologyfgvc7-keras-deeplearning/blob/master/test_images.png "Some images from the testing set")

## Proccessed images that go into model

![alt text](https://github.com/stochasticats/plantpathologyfgvc7-keras-deeplearning/blob/master/processed_image.png "Processed image")

![alt text](https://github.com/stochasticats/plantpathologyfgvc7-keras-deeplearning/blob/master/processedimage2.png "Another processed image")

![alt text](https://github.com/stochasticats/plantpathologyfgvc7-keras-deeplearning/blob/master/processedimage4.png "Another processed image")

## Model
A pre-trained MobileNet model is used.

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param  
=================================================================
mobilenet_1.00_224 (Model)   (None, 7, 7, 1024)        3228864   
_________________________________________________________________
flatten_1 (Flatten)          (None, 50176)             0         
_________________________________________________________________
dense_6 (Dense)              (None, 4)                 200708    
=================================================================
Total params: 3,429,572
Trainable params: 200,708
Non-trainable params: 3,228,864
_________________________________________________________________

Last epoch
Epoch 00043: LearningRateScheduler reducing learning rate to 1e-05.
Epoch 43/50
280/280 [==============================] - ETA: 0s - loss: 0.0364 - accuracy: 0.9911Restoring model weights from the end of the best epoch.
280/280 [==============================] - 25s 89ms/step - loss: 0.0364 - accuracy: 0.9911 - val_loss: 0.6195 - val_accuracy: 0.8514 - lr: 1.0000e-05

## Model results

![alt text](https://github.com/stochasticats/plantpathologyfgvc7-keras-deeplearning/blob/master/model_acc.png "Model accuracy over epochs")

![alt text](https://github.com/stochasticats/plantpathologyfgvc7-keras-deeplearning/blob/master/model_loss.png "Model loss over epochs")

AUC score 0.5262953416252789

Achieved a score of 0.90888 on leaderboard for the Kaggle competition.
