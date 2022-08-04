from IPython.core.display import HTML
from IPython.display import Image
from collections import Counter
import pandas as pd
import json

from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from plotly import tools
import seaborn as sns

import tensorflow as tf
import numpy as np

from PIL import Image
import urllib
from io import StringIO
from IPython.core.display import HTML



class Color(object):
    def __init__(self, filename):
        self.image_path = './data/'
        self.image_name = filename
        # self.image_ex = 'png'
        self.raw_image = self.image_path + self.image_name  # '.' + self.image_ex
        self.pre_image = self.image_path + 'processed_' + self.image_name # + '.' + self.image_ex


    def remover_background(self):
        img = Image.open(self.raw_image)
        img = img.convert("RGBA")
        datas = img.getdata()
        print(self.raw_image)
        newData = []
        cutOff = 255

        for item in datas:
            if item[0] >= cutOff and item[1] >= cutOff and item[2] >= cutOff:
                newData.append((255, 255, 255, 0))  # 배경제거
            else:
                newData.append(item)
        img.putdata(newData)
        img.save(self.pre_image, "png")  # 배경제거 이미지 저장

    def compute_average_image_color(self):
        self.img = Image.open(self.pre_image)
        rgbimg = self.img.convert('RGB')
        width, height = self.img.size
        count, r_total, g_total, b_total = 0, 0, 0, 0
        for x in range(0, width):
            for y in range(0, height):
                r, g, b = rgbimg.getpixel((x, y))
                r_total += r
                g_total += g
                b_total += b
                count += 1
        return (r_total / count, g_total / count, b_total / count)

    def discrimination_color(self):
        self.remover_background()
        average_colors = {}
        average_color = self.compute_average_image_color()
        if average_color not in average_colors:
            average_colors[average_color] = 0
        average_colors[average_color] += 1

        for average_color in average_colors:
            average_color1 = (int(average_color[0]), int(average_color[1]), int(average_color[2]))
            image_url = "<span style='display:inline-block; min-width:200px; background-color:rgb" + str(
                average_color1) + ";padding:10px 10px;'>" + str(average_color1) + "</span>"
        num_var = np.var(average_color)
        if num_var < 1:
            if average_color[0] < 200:
                return ("Black")
            else:
                return ("White")
        else:
            color_list = ['Red', 'Green', 'Blue']
            max_index = average_color.index(max(average_color))
            # print(color_list[max_index])
            return str(color_list[max_index])

    def hook(self):
        def print_menu():
            print('0. Exit')
            print('1. 이미지 전처리')
            print('2. 모델 생성(실행안해도됨)')
            print('3. 색상인식')

            return input('메뉴 선택 \n')

        while 1:
            menu = print_menu()
            if menu == '0':
                break
            elif menu == '1':
                self.remover_background()
            elif menu == '2':
                self.compute_average_image_color()
            elif menu == '3':
                self.discrimination_color()

