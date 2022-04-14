from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import *

def get_model(lr):
    
    image_input = Input(shape=(28,28,1),)

    image_rep = Conv2D(32,(5,5),)(image_input)
    image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    
    image_rep = Conv2D(128,(5,5),)(image_rep)
    image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    
    image_rep = Flatten()(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    logit = Dense(10,activation='softmax')(image_rep)
    
    model = Model(image_input,logit)
    
    model.compile(loss=['categorical_crossentropy'],
                      optimizer= SGD(lr=lr),
                      metrics=['acc'])
                      
    return model