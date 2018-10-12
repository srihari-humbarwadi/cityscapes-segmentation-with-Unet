
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf


# In[2]:


def DilatedNet(img_height, img_width, nclasses, use_ctx_module=False, bn=False):
    print('. . . . .Building DilatedNet. . . . .')
    def bilinear_upsample(image_tensor):
        upsampled = tf.image.resize_bilinear(image_tensor, size=(img_height, img_width))
        return upsampled
    
    def conv_block(conv_layers, tensor, nfilters, size=3, name='', padding='same', dilation_rate=1,pool=False):
        if dilation_rate == 1:
            conv_type = 'conv'
        else:
            conv_type = 'dilated_conv'
        for i in range(conv_layers):
            tensor = Conv2D(nfilters, size, padding=padding, use_bias=False, dilation_rate=dilation_rate, name=f'block{name}_{conv_type}{i+1}')(tensor)
            if bn:
                tensor = BatchNormalization(name=f'block{name}_bn{i+1}')(tensor)
            tensor = Activation('relu', name=f'block{name}_relu{i+1}')(tensor)
        if pool:
            tensor = MaxPooling2D(2, name=f'block{name}_pool')(tensor)
        return tensor
       
    nfilters = 64
    img_input = Input(shape=(img_height, img_width, 3))
    x = conv_block(conv_layers=2,tensor=img_input, nfilters=nfilters*1, size=3, pool=True, name=1)
    x = conv_block(conv_layers=2,tensor=x, nfilters=nfilters*2, size=3, pool=True, name=2)
    x = conv_block(conv_layers=3,tensor=x, nfilters=nfilters*4, size=3, pool=True, name=3)
    x = conv_block(conv_layers=3,tensor=x, nfilters=nfilters*8, size=3, name=4)
    x = conv_block(conv_layers=3,tensor=x, nfilters=nfilters*8, size=3,dilation_rate=2, name=5)
    x = conv_block(conv_layers=1,tensor=x, nfilters=nfilters*64, size=7,dilation_rate=4, name='_FCN1')
    x = Dropout(0.5)(x)
    x = conv_block(conv_layers=1,tensor=x, nfilters=nfilters*64, size=1, name='_FCN2')
    x = Dropout(0.5)(x)  
    x = Conv2D(filters=nclasses, kernel_size=1, padding='same', name=f'frontend_output')(x)
    if use_ctx_module:
        x = conv_block(conv_layers=2, tensor=x, nfilters=nclasses*2, size=3, name='_ctx1')
        x = conv_block(conv_layers=1, tensor=x, nfilters=nclasses*4, size=3, name='_ctx2', dilation_rate=2)
        x = conv_block(conv_layers=1, tensor=x, nfilters=nclasses*8, size=3, name='_ctx3', dilation_rate=4)
        x = conv_block(conv_layers=1, tensor=x, nfilters=nclasses*16, size=3, name='_ctx4', dilation_rate=8)
        x = conv_block(conv_layers=1, tensor=x, nfilters=nclasses*32, size=3, name='_ctx5', dilation_rate=16)        
        x = conv_block(conv_layers=1, tensor=x, nfilters=nclasses*32, size=3, name='_ctx7')
        x = Conv2D(filters=nclasses, kernel_size=1, padding='same', name=f'ctx_output')(x)
    x = Lambda(bilinear_upsample, name='bilinear_upsample')(x)
    x = Reshape((img_height*img_width, nclasses))(x)
    x = Activation('softmax', name='final_softmax')(x)
  
    model = Model(inputs=img_input, outputs=x, name='DilatedNet')
    print('. . . . .Building network successful. . . . .')
    return model


