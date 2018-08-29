from keras.models import save_model,load_model
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2


def convert_to_class_name(predictions,indices):
    """
    Convert predicted class indices to equivalent names
    """
    class_name_preds = []
    for p in predictions:
        for k in indices.keys():
            if indices[k] == p:
                class_name_preds.append(k)
    return class_name_preds

# load the stored model
model = load_model(r"E:\datasets\models\GunDet") 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range=5
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(r"E:\datasets\train_set",
                                                 target_size = (100,100),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')
pred_set=test_datagen.flow_from_directory(r'E:\datasets\test_set',
                                            target_size = (100, 100),
                                            batch_size = 10,
                                            class_mode = 'categorical')

test_image = cv2.imread(r'E:\datasets\71.jpg',1)
plt.imshow(test_image)
Xtest,Ytest = pred_set.next()
plt.imshow(Xtest[0])
convert_to_class_name(model.predict_classes(Xtest),training_set.class_indices)
model.predict_classes(Xtest)
model.predict_proba(Xtest)

