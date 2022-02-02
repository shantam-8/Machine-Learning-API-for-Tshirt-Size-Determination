import uvicorn
import numpy as np
from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
from keras.models import load_model
import tensorflow as tf

app = FastAPI()

def val_size_conversion(ans):
    if ans == 0:
        size = "XS in Short Length"
    elif ans == 1:
        size = "XS in Regular Length"
    elif ans == 2:
        size = "XS in Long Length"
    elif ans == 3:
        size = "S in Short Length"
    elif ans == 4:
        size = "S in Regular Length"
    elif ans == 5:
        size = "S in Long Length"
    elif ans == 6:
        size = "M in Short Length"
    elif ans == 7:
        size = "M in Regular Length"
    elif ans == 8:
        size = "M in Long Length"
    elif ans == 9:
        size = "L in Short Length"
    elif ans == 10:
        size = "L in Regular Length"
    elif ans == 11:
        size = "L in Long Length"
    elif ans == 12:
        size = "XL in Short Length"
    elif ans == 13:
        size = "XL in Regular Length"
    elif ans == 14:
        size = "XL in Long Length"
    elif ans == 15:
        size = "is not availabe yet"

    return(size)


#http://127.0.0.1:8000/predict
@app.get('/', response_class=HTMLResponse)
def take_inp():
    return '''
        Please input the following values. In the Standard or Relaxed Fit input box, please insert "1" for a standard fit and "0" for a relaxed fit.
        <form method="post">
        <input maxlength="5" name="height" type="number" step="0.1" placeholder="Height in cm" />
        <input maxlength="5" name="weight" type="number" step="0.1" placeholder="Weight in kg" />
        <input maxlength="5" name="shoesize" type="number" step="0.1" placeholder="EU Shoe Size" />
        <input maxlength="28" name="fit" type="number" placeholder="Standard or Relaxed Fit" />
        <input type="submit" />'''


@app.post('/')
def predict(height:float = Form(...), weight:float = Form(...), shoesize:float = Form(...), fit:float = Form(...)):
    model = tf.lite.Interpreter(model_path="Model_4.tflite")
    model.allocate_tensors()

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    val = np.array([np.array([height, weight, shoesize, fit])], dtype=np.float32)
    model.set_tensor(input_details[0]['index'], val)
    
    model.invoke()

    output_data = model.get_tensor(output_details[0]['index'])

    ans = np.argmax(output_data)
    size = val_size_conversion(ans)
    confidence = np.max(output_data) * 100

    return{f"Your perfect tshirt size is {size}. This can be said with a confidence of {confidence:.2f}%"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
