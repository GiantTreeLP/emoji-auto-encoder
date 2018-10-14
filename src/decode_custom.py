import imageio
import numpy as np

from autoencoder import get_model

if __name__ == '__main__':
    _, _, decoder = get_model()
    input_array = [0 for x in range(64)]

    prediction = decoder.predict([[np.array(input_array)]])
    print(prediction[0])
    imageio.imwrite("../test.png", prediction[0])
