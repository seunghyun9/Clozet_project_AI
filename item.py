from requests import request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


class Item(object):
    def __init__(self, filename):
        fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        self.model = self.create_model()
        self.image_path = './data/'
        self.image_name = filename
        # self.image_ex = 'png'
        self.image_data = self.image_path + self.image_name # + '.' + self.image_ex
        self.img = cv2.imread(self.image_data, 0)  # read image as gray scale

    def visualization_dataset(self):
        # 데이터 시각화
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        plt.figure(figsize=(10, 8))
        for i in range(20):
            plt.subplot(4, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[self.train_labels[i]])
        plt.show()

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.summary()

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self):
        self.model.fit(self.train_images, self.train_labels, epochs=20)
        self.model.save('./model/cloth.h5')

    def check_model(self):
        predictions = self.model.predict(self.test_images)
        predictions[0]
        np.argmax(predictions[0])
        print(np.argmax(predictions[0]))

    # 데이터 입력
    @staticmethod
    def infer_prec(img, img_size):
        img = tf.expand_dims(img, -1)  # from 28 x 28 to 28 x 28 x 1
        img = tf.divide(img, 255)  # normalize
        img = tf.image.resize(img,  # resize acc to the input
                              [img_size, img_size])
        img = tf.reshape(img,  # reshape to add batch dimension
                         [1, img_size, img_size, 1])
        return img

    def find_item(self):
        img = cv2.bitwise_not(self.img)  # < ----- bitwise_not
        print(img.shape)  # (300, 231)

        plt.imshow(img, cmap="gray")
        plt.show()

        img = self.infer_prec(img, 28)  # call preprocess function
        print(img.shape)  # (1, 28, 28, 1)

        model2 = load_model('./model/cloth.h5')
        y_pred = model2.predict(img)

        np.array([[3.1869055e-03, 5.6372599e-05, 1.1225128e-01, 2.2242602e-02,
                   7.7411497e-01, 5.8861728e-11, 8.7906137e-02, 6.2964287e-12,
                   2.4166984e-04, 2.0408438e-08]], dtype=np.float32)
        tf.argmax(y_pred, axis=-1).numpy()

        if int(tf.argmax(y_pred, axis=-1).numpy()) == 0: return ("티셔츠")
        elif int(tf.argmax(y_pred, axis=-1).numpy()) == 1 : return ("바지")
        elif int(tf.argmax(y_pred, axis=-1).numpy()) == 2 : return ("긴팔")
        elif int(tf.argmax(y_pred, axis=-1).numpy()) == 3 : return ("드레스")
        elif int(tf.argmax(y_pred, axis=-1).numpy()) == 4 : return ("코트")
        elif int(tf.argmax(y_pred, axis=-1).numpy()) == 5 : return ("샌들") 
        elif int(tf.argmax(y_pred, axis=-1).numpy()) == 6 : return ("셔츠")
        elif int(tf.argmax(y_pred, axis=-1).numpy()) == 7 : return ("스니커즈")
        elif int(tf.argmax(y_pred, axis=-1).numpy()) == 8 : return("가방")
        elif int(tf.argmax(y_pred, axis=-1).numpy()) == 9 :return ("앵클부츠")
        else : return {"item":"인식불가"}


    def hook(self):
        def print_menu():
            print('0. Exit')
            print('1. 데이터 시각화')
            print('2. 모델 생성')
            print('3. 모델 훈련')
            print('4. 모델 체크')
            print('5. 데이터 입력')
            return input('메뉴 선택 \n')

        while 1:
            menu = print_menu()
            if menu == '0':
                break
            elif menu == '1':
                self.visualization_dataset()
            elif menu == '2':
                self.create_model()
            elif menu == '3':
                self.train_model()
            elif menu == '4':
                self.check_model()
            elif menu == '5':
                self.find_item()
