import sys
import streamlit as st
import time
import cv2 as cv
import os
import csv
import re
from linedetect import crop_rect, getGrayImg
from linedecoder_mp import barDecode
import pool

import cProfile

bardet = None

def init():
    # st.freeze_support()
    print('initing')
    global bardet
    bardet = cv.barcode_BarcodeDetector()
    pool.pool = pool.init()

def decode(source):
    x1 = 0
    y1 = 0
    y2, x2 = source.shape[:2]
    # print(y2, x2)

    bar_img = source[y1:y2, x1:x2]
    # cv.imshow('asd', bar_img)
    # cv.waitKey(0)
    start_time = time.time()
    gray = cv.cvtColor(bar_img, cv.COLOR_BGR2GRAY)
    graycopy = gray.copy()
    # step 1: detect
    start_1 = time.time()
    # res, angle, rect = findLines(gray, source)
    retval, points = bardet.detect(source)
    # rect = (points[0][0], points[0][1], points[0][2])
    rect = points
    end_1 = time.time()
    # print(rect)
    if rect is None:
        return [None, None, None]

    # step 2: crop
    start_2 = time.time()
    crop = crop_rect(graycopy, rect)
    # cv.imshow('crop', crop)
    # cv.waitKey(0)
    end_2 = time.time()

    # step 3: decode
    start_3 = time.time()
    # if pic == '04_2(2)q.jpeg' or pic == '04_2(1)q.jpeg' or pic == '06_1(1)q.jpeg' or pic == '06_1(2).jpeg':
    #     visual = True
    # else:
    #     visual = False
    visual = False
    result = barDecode(crop, k=1, sig1=24, sig2=4, vis=visual)
    end_3 = time.time()
    end_time = time.time()
    exec_time = end_time - start_time

    # recalculate correction values
    if result is not None:
        corval = result[0][1] * (result[0][2] ** 2)
        sample_info = [exec_time, result[0][0], corval]
    else:
        sample_info = [exec_time, None, None]
    # info.append(sample_info)
    # print(sample_info)
    exec_1 = end_1 - start_1
    exec_2 = end_2 - start_2
    exec_3 = end_3 - start_3
    # print(exec_1, exec_2, exec_3)
    return sample_info



# if __name__ == '__main__':
#     bardet = cv.barcode_BarcodeDetector()  # initialize OpenCV barcode detector
#     dir = sys.argv[1]  # get folder path
#     samples = os.listdir(dir)  # get files from folder
#     pool.pool = pool.init()  # initialize pool for multiprocessing
#     # pool.pool2 = pool.init()
#     print(pool.pool)
#     # print(pool.pool2)
#
#     info = []  # collected data to save in csv
#
#     for pic in samples:
#         # path = '\\'.join([dir, pic])
#         path = os.path.join(dir, pic)  # path to file
#         print(path)
#         _, source = getGrayImg(path)
#
#         x1 = 0
#         y1 = 0
#         y2, x2 = source.shape[:2]
#         # print(y2, x2)
#
#         bar_img = source[y1:y2, x1:x2]
#         # cv.imshow('asd', bar_img)
#         # cv.waitKey(0)
#         start_time = time.time()
#         gray = cv.cvtColor(bar_img, cv.COLOR_BGR2GRAY)
#         graycopy = gray.copy()
#         # step 1: detect
#         start_1 = time.time()
#         # res, angle, rect = findLines(gray, source)
#         retval, points = bardet.detect(source)
#         # rect = (points[0][0], points[0][1], points[0][2])
#         rect = points
#         end_1 = time.time()
#         # print(rect)
#         if rect is None:
#             continue
#
#         # step 2: crop
#         start_2 = time.time()
#         crop = crop_rect(graycopy, rect)
#         # cv.imshow('crop', crop)
#         # cv.waitKey(0)
#         end_2 = time.time()
#
#         # step 3: decode
#         start_3 = time.time()
#         # if pic == '04_2(2)q.jpeg' or pic == '04_2(1)q.jpeg' or pic == '06_1(1)q.jpeg' or pic == '06_1(2).jpeg':
#         #     visual = True
#         # else:
#         #     visual = False
#         visual = False
#         result = barDecode(crop, k=1, sig1=24, sig2=4, vis=visual)
#         end_3 = time.time()
#         end_time = time.time()
#         exec_time = end_time - start_time
#
#         # recalculate correction values
#         if result is not None:
#             corval = result[0][1] * (result[0][2] ** 2)
#             sample_info = [pic, exec_time, result[0][0], corval]
#         else:
#             sample_info = [pic, exec_time, None, None]
#         info.append(sample_info)
#         print(sample_info)
#         exec_1 = end_1 - start_1
#         exec_2 = end_2 - start_2
#         exec_3 = end_3 - start_3
#         print(exec_1, exec_2, exec_3)
#
#         # cv.imshow('res', res)
#         # cv.waitKey(1)
#
#     with open('result1.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile, delimiter=' ',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         writer.writerows(info)

