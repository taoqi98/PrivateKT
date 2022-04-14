import numpy as np 

def evaluation(model,test_images,test_labels):
    pred = model.predict(test_images).argmax(axis=-1)
    labels = test_labels.argmax(axis=-1)
    return (pred==labels).mean()