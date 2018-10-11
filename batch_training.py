import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.python.keras.optimizers import Adadelta, Nadam
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import multi_gpu_model, plot_model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
from multiclassunet import Unet
import tqdm
import cv2
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.callbacks import Callback
from dilatednet import DilatedNet

# In[2]:


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss


# In[3]:


cats = {'void': [0, 1, 2, 3, 4, 5, 6],
 'flat': [7, 8, 9, 10],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}


# In[4]:


image_dir = 'aug_dataset/images'
mask_dir = 'aug_dataset/masks'
image_list = os.listdir(image_dir)
mask_list = os.listdir(mask_dir)
image_list.sort()
mask_list.sort()
print(f'. . . . .Number of images: {len(image_list)}\n. . . . .Number of masks: {len(mask_list)}')

# sanity check
for i in range(len(image_list)):
    assert image_list[i][16:] == mask_list[i][24:]


batch_size = 16
samples = 50000
steps = samples//batch_size
img_height, img_width = 256, 256
classes = 8
filters_n = 64

class seg_gen(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        idx = np.random.randint(0, 50000, batch_size)
        batch_x, batch_y = [], []
        drawn = 0
        for i in idx:
            _image = image.img_to_array(image.load_img(f'{image_dir}/{image_list[i]}', target_size=(img_height, img_width)))/255.   
            img = image.img_to_array(image.load_img(f'{mask_dir}/{mask_list[i]}', grayscale=True, target_size=(img_height, img_width)))
            labels = np.unique(img)
            if len(labels) < 3:
                idx = np.random.randint(0, 50000, batch_size-drawn)
                continue
            img = np.squeeze(img)
            mask = np.zeros((img.shape[0], img.shape[1], 8))
            for i in range(-1, 34):
                if i in cats['void']:
                    mask[:,:,0] = np.logical_or(mask[:,:,0],(img==i))
                elif i in cats['flat']:
                    mask[:,:,1] = np.logical_or(mask[:,:,1],(img==i))
                elif i in cats['construction']:
                    mask[:,:,2] = np.logical_or(mask[:,:,2],(img==i))
                elif i in cats['object']:
                    mask[:,:,3] = np.logical_or(mask[:,:,3],(img==i))
                elif i in cats['nature']:
                    mask[:,:,4] = np.logical_or(mask[:,:,4],(img==i))
                elif i in cats['sky']:
                    mask[:,:,5] = np.logical_or(mask[:,:,5],(img==i))
                elif i in cats['human']:
                    mask[:,:,6] = np.logical_or(mask[:,:,6],(img==i))
                elif i in cats['vehicle']:
                    mask[:,:,7] = np.logical_or(mask[:,:,7],(img==i))
            mask = np.resize(mask,(img_height*img_width, 8))
            batch_y.append(mask)
            batch_x.append(_image)
            drawn += 1
        return np.array(batch_x), np.array(batch_y)


class visualize(Callback):
    def on_epoch_end(self, epoch, logs):
        print(f'\nGenerating output at epoch : {epoch}')
        i = 567
        img = image.img_to_array(image.load_img(f'{image_dir}/{image_list[i]}'))/255.    
        dims = img.shape
        x = cv2.resize(img, (256, 256))
        x = np.float32(x)/255.
        z = unet.predict(np.expand_dims(x, axis=0))
        z = np.squeeze(z)
        z = z.reshape(256, 256, 8)
        z = cv2.resize(z, (dims[1], dims[0]))
        y = np.argmax(z, axis=2)
        
        construction = np.zeros_like(y)
        human = np.zeros_like(y)
        vehicle = np.zeros_like(y)
        construction[y==2] = 255.
        human[y==6] = 255.
        vehicle[y==7] = 255.
        
        result = img.copy()
        alpha = 0.4
        img[:,:,1] = construction
        img[:,:,2] = vehicle 
        img[:,:,0] = human

        cv2.addWeighted(img, alpha, result, 1-alpha, 0, result)
        cv2.imwrite(f'outputs/{epoch}.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print('Wrote file to disk')

unet = DilatedNet(256, 256, 8,use_ctx_module=True, bn=True)
p_unet = multi_gpu_model(unet, 4)
p_unet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[dice_coeff, 'accuracy'])
tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(mode='max', filepath='models-dr/pdilated.h5', monitor='acc', save_best_only='True', save_weights_only='True', verbose=1)
es = EarlyStopping(mode='max', monitor='acc', patience=6, verbose=1)
vis = visualize()
callbacks = [tb, mc, es]
train_gen = seg_gen(image_list, mask_list, batch_size)


p_unet.fit_generator(train_gen, steps_per_epoch=steps, epochs=8, callbacks=callbacks, workers=8)
print('Saving final weights')
unet.save_weights('dilated.h5')

