# Machine Learning API for Tshirt Size Determination
## Introduction
7.8 billion different people but just five conventional t-shirt sizes – XS, S, M, L, and XL. Hence, to provide a much better fit, we are creating three different t-shirt lengths (the Short, Regular, and Long) for each conventional t-shirt size – effectively providing you with fifteen different options. The following table clearly explains the sizes provided.
| Dimensions (in CM) | XS | S | M | L | XL |
| ------ | ------ | ------ | ------ | ------ | ------ |
| Shoulder to Shoulder | 42.5 |44.5 |46.5 |48.5 |50.5 |
| Chest | 46.0 |49.0 |52.0 |55.0 |58.0 |
| Waist | 44.0 |47.0 |50.0 |53.0 |56.0 |
| Bottom | 46.0 |49.0 |52.0 |55.0 |58.0 |
| **Short Length** |60.0 |63.0 |66.0 |69.0 |72.0 |
| **Regular Length** | 63.0 |66.0 |69.0 |72.0 |75.0 |
| **Long Length** | 66.0 |69.0 |72.0 |75.0 |78.0 |

To further ease this process, the following Neural Network powered API takes the user’s height, weight, shoe size, and preferred t-shirt fit (Standard or Relaxed) and provides the perfect t-shirt with its respective confidence percentage.

Please note that this API is currently in its development stage and can only consider men’s sizes. Future developments include reformatting the front end of the API and improving the model to provide more accurate t-shirt sizes.

## Visuals
The following visual illustrates the working of the API.

![Working of API](https://user-images.githubusercontent.com/64306405/134773551-00e4576b-a5b5-4d32-acd6-b4285bb77b5c.gif)

## Technology
### Model Construction
The model [(Model_4.h5)](https://github.com/shantam-8/Machine-Learning-API-for-Tshirt-Size-Determination/blob/main/Model_4.h5) is a Neural Network constructed using the TensorFlow Library. After several evaluations it was adjudged that the “swish” activation increased the accuracy of the model when compared with standard activations like “ReLU”. The ["swish" activation](https://www.geeksforgeeks.org/ml-swish-function-by-google-in-keras/) was used in the following manner.

```python
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': swish})

#Replacing "ReLU" with "swish" in the activation parameter of Sequential. 
import tensorflow as tf

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Dense(6000, activation="swish"),
                                    ])
```
Additionally, of the 500 data points sourced from different websites, 70% were used as the training set and 30% were used as the test set. Through several processes of trial-and-error, a Training Accuracy of 0.9521 and a Test Accuracy of 0.8541 was achieved.
