# pre-trained Model

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.layers import Dense, Flatten

# model.trainable = False
models = [
    VGG16(), VGG19(), Xception(), 
    DenseNet121(), DenseNet169(), DenseNet201(),
    ResNet50(), ResNet50V2(),
    ResNet101(), ResNet101V2(), ResNet152(), ResNet152V2(),
    InceptionV3(), InceptionResNetV2(),
    MobileNet(), MobileNetV2(), MobileNetV3Large(), MobileNetV3Small(),
    NASNetLarge(), NASNetMobile(),
    EfficientNetB0(), EfficientNetB1(), EfficientNetB7(),
]

for model in models:
    model.trainable = False
    # if model.trainable_weights != 0:
    #     trainable_p = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    # else:
    #     trainable_p = 0
    # if model.non_trainable_weights != 0:
    #     non_trainable_p = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    # else:
    #     non_trainable_p = 0
    # model.summary()
    print(f'********** {model.name} **********')
    print('Total params: ', model.count_params())
    # print('Trainable params: ', int(np.sum([K.count_params(p) for p in set(model.trainable_weights)])))
    # print('Non_trainable params: ', int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])))
    print('전체 가중치 개수: ', len(model.weights))
    print('훈련 가능한 가중치 개수: ', len(model.trainable_weights))

# model = VGG16()
# model.non_trainable_weights

# 모델별 파라미터와 웨이트 수 정리하기
'''
***** vgg16 *****
Total params:  138357544   
전체 가중치 개수:  32      
훈련 가능한 가중치 개수:  0
***** vgg19 *****
Total params:  143667240   
전체 가중치 개수:  38      
훈련 가능한 가중치 개수:  0
***** xception *****       
Total params:  22910480    
전체 가중치 개수:  236     
훈련 가능한 가중치 개수:  0
***** densenet121 *****    
Total params:  8062504     
전체 가중치 개수:  606     
훈련 가능한 가중치 개수:  0
***** densenet169 *****    
Total params:  14307880    
전체 가중치 개수:  846     
훈련 가능한 가중치 개수:  0
***** densenet201 *****
Total params:  20242984    
전체 가중치 개수:  1006    
훈련 가능한 가중치 개수:  0
***** resnet50 *****       
Total params:  25636712    
전체 가중치 개수:  320
훈련 가능한 가중치 개수:  0
***** resnet50v2 *****
Total params:  25613800
전체 가중치 개수:  272
훈련 가능한 가중치 개수:  0
***** resnet101 *****
Total params:  44707176
전체 가중치 개수:  626
훈련 가능한 가중치 개수:  0
***** resnet101v2 *****
Total params:  44675560
전체 가중치 개수:  544
훈련 가능한 가중치 개수:  0
***** resnet152 *****
Total params:  60419944
전체 가중치 개수:  932
훈련 가능한 가중치 개수:  0
***** resnet152v2 *****
Total params:  60380648
전체 가중치 개수:  816
훈련 가능한 가중치 개수:  0
***** inception_v3 *****
Total params:  23851784
전체 가중치 개수:  378
훈련 가능한 가중치 개수:  0
***** inception_resnet_v2 *****
Total params:  55873736
전체 가중치 개수:  898
훈련 가능한 가중치 개수:  0
***** mobilenet_1.00_224 *****
Total params:  4253864
전체 가중치 개수:  137
훈련 가능한 가중치 개수:  0
***** mobilenetv2_1.00_224 *****
Total params:  3538984
전체 가중치 개수:  262
훈련 가능한 가중치 개수:  0
***** MobilenetV3large *****
Total params:  5507432
전체 가중치 개수:  266
훈련 가능한 가중치 개수:  0
***** MobilenetV3small *****
Total params:  2554968
전체 가중치 개수:  210
훈련 가능한 가중치 개수:  0
***** NASNet *****
Total params:  88949818
전체 가중치 개수:  1546
훈련 가능한 가중치 개수:  0
***** NASNet *****
Total params:  5326716
전체 가중치 개수:  1126
훈련 가능한 가중치 개수:  0
***** efficientnetb0 *****
Total params:  5330571
전체 가중치 개수:  314
훈련 가능한 가중치 개수:  0
***** efficientnetb1 *****
Total params:  7856239
전체 가중치 개수:  442
훈련 가능한 가중치 개수:  0
***** efficientnetb7 *****
Total params:  66658687
전체 가중치 개수:  1040
훈련 가능한 가중치 개수:  0
'''