#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from PIL import Image, ImageDraw, ImageFont
'''
Author:     Yichen Nie
Ref:        https://blog.csdn.net/five3/article/details/82464723
Date created:11/23/2020
'''

def draw_num(new_img, num, font, fc):
    num = str(num)
    fc = tuple(fc)
    draw = ImageDraw.Draw(new_img)
    img_size = new_img.size
    font_size = 20
    fnt = ImageFont.truetype(font, font_size)
    fnt_size = fnt.getsize(num)
    x = (img_size[0] - fnt_size[0]) / 2
    y = (img_size[1] - fnt_size[1]) / 2
    draw.text((x, y), num, font=fnt, fill=fc)


def file_name(file_dir, typ):
    files=[]
    # root = os.path.split(os.path.realpath(__file__))[0]
    for file in os.listdir(file_dir):
        if os.path.splitext(file)[1] == '.' + typ:
            files.append(file)
            # L.append(os.path.join(root, file))
    return files


def savepng(num, font, fc, bc):
    """
    Generate a picture with number `num`.\\
    Font: `font`.\\
    Font color: `fc`.\\
    Background color: `bc`.
    """
    fc = tuple(fc)
    bc = tuple(bc)
    img = Image.new("RGB", (28, 28), color=(0, 0, 0))
    draw_num(img, num, font, fc)
    img = img.convert('L') # 灰度图
    filename = '../data/numgen/num%s_%s_Rb%sGb%sBb%sRf%sGf%sBf%s.png' % tuple([num, font] + list(bc) + list(fc))
    img.save(filename)

 
def main():
    # font = ['Arial.ttf', 'Menlo.ttc', 'Symbol.ttf']
    font_folder = '/System/Library/Fonts'
    ttf = file_name(font_folder, 'ttf')
    ttc = file_name(font_folder, 'ttc')
    otf = file_name(font_folder, 'otf')
    font = ttf + ttc + otf

    data_folder ='../data/numgen/'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    
    for i in range(1000):
        number = random.randint(0, 9)
        font_label = random.randint(0, len(font)-1)
        fontcolor = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        backcolor = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        savepng(number, font[font_label], fontcolor, backcolor)

if __name__ == '__main__':
    main()
