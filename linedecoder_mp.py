# barcode decoding from cropped and rotated image

#   barDecode()
#       imgProc()
#           bluring
#           local_norm() normilization
#       readCode()
#           get multiple lines of barcode
#           for each line
#               extStretch() stretching uses extDetect()
#           multiprocessed oneThreshScan() for each stretched lines and each thresholds
#           for results of oneThreshScan()
#               get edges with different thresholds
#               transform edges to widthes (units) with symScan()
#               voting() getting most possible widthes
#               infodecoder() decode it
#           collect all decoded and choose one

# test() perform barDecode()
# checkFolder() do test() for whole directory
# check() same as checkFolder() but also checks answers


import cv2 as cv
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from infodecoder import infodecoder
from os import listdir
import copy
import math
import pool



NSTEP = 8  # number of line per barcode, less -> faster
STEP = 16  # step of in checking threshold in lines, less -> slower
BLUR_VAL = 25  # value used in blur through y axis, less -> stronger blur

# answers for 120crop set
ans = {
    '01_1(1).jpeg': '<FNC1>403DE9461890',
    '01_1(2).jpeg': '<FNC1>00459632452204823586',
    '01_2(1).jpeg': '<FNC1>403DE9461890',
    '01_2(2).jpeg': '<FNC1>00459632452204823586',
    '01_3(1).jpeg': '<FNC1>403DE9461890',
    '01_3(2).jpeg': '<FNC1>00459632452204823586',
    '02_1(1).jpeg': '<FNC1>403DE123961',
    '02_1(2).jpeg': '<FNC1>00182189061678946020',
    '02_2(1).jpeg': '<FNC1>403DE123961',
    '02_2(2).jpeg': '<FNC1>00182189061678946020',
    '02_3(1).jpeg': '<FNC1>403DE123961',
    '02_3(2).jpeg': '<FNC1>00182189061678946020',
    '03_1(1).jpeg': '<FNC1>403DE394817',
    '03_1(2).jpeg': '<FNC1>00378250486287288431',
    '03_2(1).jpeg': '<FNC1>403DE394817',
    '03_2(2).jpeg': '<FNC1>00378250486287288431',
    '03_3(1).jpeg': '<FNC1>403DE394817',
    '03_3(2).jpeg': '<FNC1>00378250486287288431',
    '04_1(1).jpeg': '<FNC1>403DE98801',
    '04_1(2).jpeg': '<FNC1>00569954285165295049',
    '04_2(1).jpeg': '<FNC1>403DE98801',
    '04_2(2).jpeg': '<FNC1>00569954285165295049',
    '04_3(1).jpeg': '<FNC1>403DE98801',
    '04_3(2).jpeg': '<FNC1>00569954285165295049',
    '05_1(1).jpeg': '<FNC1>403DE728199',
    '05_1(2).jpeg': '<FNC1>00596376979304477659',
    '05_2(1).jpeg': '<FNC1>403DE728199',
    '05_2(2).jpeg': '<FNC1>00596376979304477659',
    '05_3(1).jpeg': '<FNC1>403DE728199',
    '05_3(2).jpeg': '<FNC1>00596376979304477659',
    '06_1(1).jpeg': '<FNC1>403DE4019485',
    '06_1(2).jpeg': '<FNC1>00846143568214029166',
    '06_2(1).jpeg': '<FNC1>403DE4019485',
    '06_2(2).jpeg': '<FNC1>00846143568214029166',
    '06_3(1).jpeg': '<FNC1>403DE4019485',
    '06_3(2).jpeg': '<FNC1>00846143568214029166',
    '07_1(1).jpeg': '<FNC1>403DE41876',
    '07_1(2).jpeg': '<FNC1>00525403852481395073',
    '07_2(1).jpeg': '<FNC1>403DE41876',
    '07_2(2).jpeg': '<FNC1>00525403852481395073',
    '07_3(1).jpeg': '<FNC1>403DE41876',
    '07_3(2).jpeg': '<FNC1>00525403852481395073',
}


def getGrayImg(path):
    """
    Read image and return grayscaled.
    :param path: path to file
    :return: gray image
    """
    try:
        source = cv.imread(path)
        if source is None:
            raise Exception()
    except:
        print("Error reading")
        return None

    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    return gray


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def local_norm(img, sigma_1X, sigma_1Y, sigma_2X, sigma_2Y):
    """Return a normilized version of the image, using a Gaussian blur.
    :param img: grayscale image
    :return: normilized image
    """
    # algorithm from http://bigwww.epfl.ch/sage/soft/localnormalization/
    # demo http://bigwww.epfl.ch/demo/ip/demos/local-normalization/
    float_gray = img.astype(np.float32) / 255.0

    blur = cv.GaussianBlur(float_gray, (0, 0), sigmaX=sigma_1X, sigmaY=sigma_1Y)  # estimation of a local mean
    # blur = cv.GaussianBlur(float_gray, (0, 0), sigmaX=sigma_1, sigmaY=sigma_1)
    num = float_gray - blur

    if sigma_2X + sigma_2Y != 0:
        blur = cv.GaussianBlur(num * num, (0, 0), sigmaX=sigma_2X, sigmaY=sigma_2Y)  # estimation of local variance
        # blur = cv.GaussianBlur(num * num, (0, 0), sigmaX=sigma_2, sigmaY=sigma_2)
        den = cv.pow(blur, 0.5)

    gray = num / den

    cv.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)

    return gray * 255


def imgProc(gray, sig1=32, sig2=1, k=1, vis=False):
    """
    All image transforms together
    :param gray: gray image
    :param sig1: parameter of normilization
    :param sig2: parameter of normilization
    :param k: coefficient for upscale
    :param vis: if need to show image
    :return: image after transformations
    """
    # currently k=1 everywhere, upscale moved to cropping

    # resize, corrently doesn't do anything
    h, w = gray.shape
    gray = cv.resize(gray, (w * k, h * k), interpolation=cv.INTER_LINEAR)
    h, w = gray.shape

    # show resized image
    if vis:
        cv.imshow('source', gray)
        cv.waitKey(0)

    # blur
    # blur = cv.GaussianBlur(gray, (0, 0), sigmaX=h / 250, sigmaY=h / BLUR_VAL)  #
    blur = cv.GaussianBlur(gray, (0, 0), sigmaX=h / 250, sigmaY=h / 75)

    # sharpening (flipping is not necessary)
    # sharp = 255 - sharp
    # sharp = unsharp_mask(sharp, (5, 5), 3.0, 1.5, 32)
    # sharp = 255 - sharp
    # sharp = unsharp_mask(sharp, (5, 5), 3.0, 1.5, 32)

    # skip sharpening step
    sharp = blur

    # local normalization
    # all constants in parameters should be recalculated
    # norm = local_norm(sharp, 276 * 8 / (512 / sig1), 276 / (512 / sig1), 276 / 2 / (512 / sig2), 276 * 8 / (512 / sig2))
    # norm = local_norm(sharp, 100, 1, 2.5, 1)
    norm = local_norm(sharp, h / 3, 1, h / 125, 1)
    # norm = local_norm(sharp, h / 3, 1, h / 75, 1)
    # norm = sharp

    # blur2 = cv.GaussianBlur(norm, (0, 0), sigmaX=h / 750, sigmaY=h / 100)

    # erode = norm.copy()
    # kernel = np.ones((h//25, 1), np.uint8)
    # erode = cv.erode(norm, kernel=kernel, iterations=1)

    # show normalized image
    if vis:
        cv.imshow('norm', np.uint8(norm))
        # cv.imshow('blur2', np.uint8(blur2))
        # cv.imshow('erode', np.uint8(erode))
        cv.waitKey(0)

    return norm


def extDetect(line, threshold=10):
    """Detect points for stretching.
    :param line: line of barcode
    :param threshold: threshold to ignore small peaks
    :return: points for stretching
    """
    white = True  # start color is whitespace
    ext = []
    lastval = line[0]  # last value (or line[0] for start) of ext point
    lastidx = 0  # last idx (or 0 for start) of ext point

    # go through line, and look for ext depending on and switching 'white' if necessary
    for idx, val in enumerate(line):
        if white:
            if val > lastval:
                lastval = val
                lastidx = idx
            if val < lastval - threshold:
                white = False
                ext.append(lastidx)
                lastval = val
                lastidx = idx
        else:
            if val < lastval:
                lastval = val
                lastidx = idx
            if val > lastval + threshold:
                white = True
                ext.append(lastidx)
                lastval = val
                lastidx = idx

    if white:
        ext.append(lastidx)
    return ext


def extStretch(line, threshold=10, a=0, b=255):
    """Stretching for separating colors.
    :param line: line of barcode
    :param threshold: threshold to ignore small peaks
    :param a: lowest value for stretching
    :param b: hightest value for stretching
    :return: stretched line
    """
    ext = extDetect(line, threshold)
    result = line.copy()

    # first range stretching
    for i in range(ext[0]):
        result[i] = line[i] + b - line[ext[0]]

    # middle ranges stretching
    for i in range(len(ext) - 1):
        alt = abs(line[ext[i]] - line[ext[i+1]])
        low = min(line[ext[i+1]], line[ext[i]])
        for j in range(ext[i], ext[i+1]):
            result[j] = (line[j] - low) * (b - a) / alt + a

    # last range stretching
    for i in range(ext[-1], len(line), 1):
        result[i] = line[i] + b - line[ext[-1]]

    return result


def symScan(edges):
    """Return bars widthes in units
    :param edges: edges of bars of one symbol code
    :return: bars widthes in units
    """
    # edges represented in float values
    totalwidth = edges[6] - edges[0]
    singlew = totalwidth / 11
    width = [round((edges[i + 1] - edges[i]) / singlew) for i in range(6)]
    return width.copy()


def is_correct(symcode):
    """Check if code of symbol is correct
    :param symcode: widthes in units
    :return: if code of symbol is correct
    """
    code = [max(min(val, 4), 1) for val in symcode]
    if sum(code) != 11:
        return False
    if sum(code[::2]) % 2 != 0:
        return False
    return True


def voting(linesymcodes):
    """Find the most likely code, after reading multiple lines"""
    result = []

    # Can add filter for lines before voting
    # goodcodes = []
    # for linecode in linesymcodes:
    #     if linecode[0][0] == 2:
    #         goodcodes.append(linecode)
    # print('goodcodes', len(goodcodes))

    # skip filtering
    goodcodes = linesymcodes.copy()

    # check if codes have same length, needed to do voting properly
    samelength = True
    for line in goodcodes:
        if len(line) != len(goodcodes[0]):
            samelength = False
            break

    if not samelength:
        return

    for symidx in range(len(goodcodes[0])):

        vote = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]  # vote frequency
        for line in goodcodes:
            code = line[symidx]
            if not is_correct(code):
                continue
            for idx, val in enumerate(code):
                # add width in vote
                w = max(min(val, 4), 1) - 1
                vote[idx][w] += 1

        # find results of voting
        des = []
        for dens in vote:
            des.append(dens.index(max(dens))+1)
        result.append(des)
    return result


def oneThreshScan(args):
    line = args[0]
    threshold = args[1]
    edges = []
    white = line[0] > threshold  # if we above of threshold

    for i in range(len(line)):
        if white:
            if line[i] < threshold:
                edge = i - (threshold - int(line[i])) / (int(line[i - 1]) - int(line[i]))
                edges.append(edge)
                white = not white
        else:
            if line[i] > threshold:
                edge = i - (threshold - int(line[i])) / (int(line[i - 1]) - int(line[i]))
                edges.append(edge)
                white = not white
    return edges.copy()


def lineThreshScan(line, mid=128, radius=96, step=1):
    """Return bar widthes from multiple thresholds lines
    :param line: line of barcode
    :param mid: middle of range for threshold
    :param radius: radius of range for threshold
    :param step: step of range
    :return: decoded code
    """

    # get intersection with threshold line
    # in other terms intersect interpolated line intensity with threshold, so obtained edges are float
    func_args = [[line, threshold] for threshold in range(mid - radius, mid + radius, step)]
    # lineres = [0] * len(func_args)

    lineres = pool.pool.map(oneThreshScan, func_args)

    # transform all edges to symcodes in units
    linesymcodes = []
    for lineedge in lineres:
        linesymres = []
        for i in range(0, len(lineedge) - 6, 6):
            symcode = symScan(lineedge[i:i + 7])
            linesymres.append(symcode.copy())
        linesymcodes.append(linesymres.copy())

    # perform voting
    width = voting(linesymcodes)

    # skip cheking
    correct = True

    # for sym in width:
    #     if not is_correct(sym):
    #         correct = False
    #         break

    if correct and (width is not None):
        return infodecoder(width, False)
    else:
        return None


def lineReadCode(line):
    stretch = extStretch(line, threshold=15)
    decoded = lineThreshScan(stretch, step=STEP)
    return decoded


def readCode(norm):
    """
    Take multiple lines from image and try to read them
    :param norm:
    :return: (
        readed and decoded code,
        correction_value1,
        correction_value2
    )
    """
    h = norm.shape[0]

    # range for taking lines
    a = h / 4
    b = h * 3 / 4
    nstep = NSTEP
    nline = [math.floor(a + (b - a) / (nstep - 1) * i) for i in range(nstep)]

    mid = 128
    radius = 96
    step = STEP
    sublen = math.ceil((radius * 2) / step)
    func_args = [] #  [[line, threshold] for threshold in range(mid - radius, mid + radius, step)]
    countread = 0
    for y in nline:
        stretch = extStretch(norm[y], threshold=25)
        func_args += [[stretch, threshold] for threshold in range(mid - radius, mid + radius, step)]


    lineres = pool.pool.map(oneThreshScan, func_args)
    readed = {}
    for i in range(len(lineres) // sublen):
        subres = lineres[i*sublen:(i+1)*sublen]
        linesymcodes = []
        for lineedge in subres:
            linesymres = []
            for i in range(0, len(lineedge) - 6, 6):
                symcode = symScan(lineedge[i:i + 7])
                linesymres.append(symcode.copy())
            linesymcodes.append(linesymres.copy())

        # perform voting
        width = voting(linesymcodes)

        # skip cheking
        correct = True

        # for sym in width:
        #     if not is_correct(sym):
        #         correct = False
        #         break

        if correct and (width is not None):
            decoded =  infodecoder(width, False)
        else:
            decoded = None
        # decoded = lineThreshScan(stretch, step=STEP)
        if decoded is not None:
            countread += 1
            # return decoded
            # print(decoded)
            # print(h//2 - h//4, y, h//2 + h//4)
            if decoded in readed.keys():
                readed[decoded] += 1
            else:
                readed.update({decoded: 1})

    # lines = [norm[y] for y in nline]
    #
    # print(pool.pool)
    # print(pool.pool2)
    # decodedlines = pool.pool2.map(lineReadCode, lines)
    # for decoded in decodedlines:
    #     if decoded is not None:
    #         countread += 1
    #         if decoded in readed.keys():
    #             readed[decoded] += 1
    #         else:
    #             readed.update({decoded: 1})

    if len(readed) == 0:
        return None

    return (max(readed, key=readed.get), max(readed.values()) / nstep, max(readed.values()) / countread)


def barDecode(gray, sig1=32, sig2=1, k=1, vis=False):
    """
    Perform barcode decoding
    :param gray: grayscale image
    :param sig1: parameter of normilization
    :param sig2: parameter of normilization
    :param k: coefficient for upscale
    :param vis: if need to show image
    :return: (
        decoded value with correction values,
        is_code_accurate_enough
    )
    """
    # do image transformations
    norm = imgProc(gray, sig1, sig2, k, vis)

    # read code from transformated image
    result = readCode(norm)

    # print(result)

    # try to check correction values
    coredge = 0.5 ** 3
    if result is not None:
        corval = result[1] * (result[2] ** 2)
        if corval > coredge:
            # print('Correct {:4.2f}'.format(corval))
            return result, True
        else:
            # print('Incorrect {:4.2f}'.format(corval))
            return result, False
    return None


def test(path, vis=False):
    """
    Get image from path and try read code.
    :param path: path to file
    :param vis: if need to show image
    :return:
    """
    gray = getGrayImg(path)
    res = barDecode(gray, k=1, sig1=24, sig2=4, vis=vis)
    return res


def checkFolder(dir, vis=False):
    """
    Do test() on all files in directory
    :param dir: directory
    :param vis: if need to show image
    """
    samples = listdir(dir)
    score = 0  # succeses depending on correction values
    total = 0  # total number of barcodes
    for pic in samples:
        path = '\\'.join([dir, pic])
        print('Begin {}'.format(path))
        res = test(path, vis)
        total += 1
        if res is not None:
            if res[1]:
                score += 1
        # print(res)

    print(total, score)


def check(dir, answers, vis=False):
    """
    Do test() on all files in directory and compare it with answers
    :param dir: directory
    :param answers: right answers for barcodes
    :param vis: if need to show image
    """
    score = 0  # succeses depending on answers
    total = 0  # total number of barcodes
    corlist = []
    for file, code in answers.items():
        total += 1
        path = '\\'.join([dir, file])
        print('Begin {}'.format(path))
        res = test(path, vis)
        if res is None:
            continue
        corval = res[0][1] * (res[0][2] ** 2)
        corlist.append(float(corval))
        if res is not None:
            if res[0][0] == code:
                print('Correct')
                score += 1
            else:
                print('Incorrect')

    print(total, score)
    corlist.sort()

    # display sorted list of correction values
    print(corlist)


if __name__ == '__main__':
    e1 = cv.getTickCount()

    # test('100crop\\CrpIMG_20220524_180429(2).jpg')
    # checkFolder('dist\\100cropRotated')
    # checkFolder('sampleRotated')
    # checkFolder('dist\\100cropRotated', vis=False)
    # test('dist\\cropRotated\\03_2(1).jpeg', vis=True)
    # test('dist\\cropRotated\\07_2(2).jpeg', vis=True)

    # test('dist\\100cropRotated\\05_2(1).jpeg', vis=True) # 0.81
    # test('dist\\100cropRotated\\01_2(1).jpeg', vis=True) # 0.50
    # test('dist\\100cropRotated\\04_3(2).jpeg', vis=True) # 0.88


    # test('dist\\100cropRotated\\02_2(1).jpeg', vis=True) # 0.81

    # check('dist\\100cropRotated', ans, vis=True)
    # check('dist\\75cropRotated', ans, vis=False)
    # check('dist\\120cropRotated', ans, vis=False)
    # check('sample', ans, vis=False)


    # test('dist\\120cropRotated\\04_1(2).jpeg', vis=True) # 0.88



    e2 = cv.getTickCount()
    time = (e2 - e1) / cv.getTickFrequency()
    print('total', time)

