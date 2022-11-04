# cropping and rotating barcode

from os import listdir
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math


def getGrayImg(path):
    """
    Read image and return it with grayscaled copy.
    :param path: path to file
    :return: (
        gray image,
        source image
    )
    """
    try:
        source = cv.imread(path)
        if source is None:
            raise Exception()
    except:
        print("Error reading")
        return None

    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    return gray, source

# def findLines(gray, source):
#     """
#     detect rectangle with same oriented lines from gray image
#     draw detected lines on source image
#     :param gray: gray image
#     :param source: source image
#     :return: (
#         edited source image,
#         detected line angle,
#         detected rotated rectangle
#     )
#     """
#     h, w = source.shape[:2]  # dimensions of image
#
#     # Detect lines in the image
#     lsd = cv.createLineSegmentDetector(0)
#     segments = lsd.detect(gray)[0]  # Position 0 of the returned tuple are the detected lines
#
#     # Draw detected lines in the image
#     # drawn_img = lsd.drawSegments(gray, segments)
#     # print(segments[0][0])
#
#     sx = []  # Xcoordinate of segments as vectors
#     sy = []  # Ycoordinate of segments as vectors
#     for segment in segments:
#         sx.append(segment[0, 2] - segment[0, 0])
#         sy.append(segment[0, 3] - segment[0, 1])
#
#     npsx = np.array(sx)
#     npsy = np.array(sy)
#     nplen = np.sqrt(np.power(sx, 2) + np.power(sy, 2))  # lengthes of segments
#     npdir = np.arctan2(npsy, npsx) % (np.pi)  # directions of segments
#
#     bins = 360  # number of bins in histogram, actually pi divided by bins
#
#     # create histogram and find max pos, that will be angle of barcode
#     y, x, patches = plt.hist(npdir, bins, weights=nplen, range=[0, np.pi])
#     angle = x[np.where(y == y.max())][0]
#     angle += np.pi/(bins*2)  # add half of bin's width because of offset
#
#     # find length of longest segment
#     maxl = 0
#     for idx, segment in enumerate(segments):
#         if abs(abs(npdir[idx] - angle) - np.pi / 2) >= np.pi / 2 * 0.95:
#             if (nplen[idx] > maxl):
#                 maxl = nplen[idx]
#
#     pad = 5  # padding near edges where lines doesnt counts
#     points = [[w / 2, h / 2]]  # points of segments with angle close to angle of barcode,
#     # [w / 2, h / 2] is center point, it must be inside of barcode
#
#
#     for idx, segment in enumerate(segments):
#         pt1 = (round(segment[0, 0]), round(segment[0, 1]))
#         pt2 = (round(segment[0, 2]), round(segment[0, 3]))
#         notonedge1 = pt1[0] > pad and pt1[1] > pad and pt1[0] < w-pad and pt1[1] < h-pad  # 1st point not near edge
#         notonedge2 = pt2[0] > pad and pt2[1] > pad and pt2[0] < w-pad and pt2[1] < h-pad  # 2nd point not near edge
#         sameangle = abs(abs(npdir[idx] - angle) - np.pi/2) >= np.pi/2 * 0.95  # segment direction close to barcode direction
#
#         if sameangle and nplen[idx] > maxl * 0.2 and notonedge1 and notonedge2:
#             # add pt1 and pt2 to points, and set draw color to green
#             # color = (0, 255, 0)
#             points.append([segment[0, 0], segment[0, 1]])
#             points.append([segment[0, 2], segment[0, 3]])
#         # else:
#             # set draw color to blue
#             # color = (255, 0, 0)
#         # draw line on source
#         # cv.line(source, pt1, pt2, color, 1)
#
#     # draw red line perpendicular to barcode through center of image
#     # drawOrient(angle, source)
#
#     # prepare point to rotation
#     points = np.array(points)
#
#     rotAngle = np.pi / 2 - angle
#     # rotation matrix
#     rot = np.array([
#         [np.cos(rotAngle), -np.sin(rotAngle)],
#         [np.sin(rotAngle), np.cos(rotAngle)],
#     ])
#     # reverse rotation matrix
#     rotrev = np.array([
#         [np.cos(-rotAngle), -np.sin(-rotAngle)],
#         [np.sin(-rotAngle), np.cos(-rotAngle)],
#     ])
#
#     # perform rotaion
#     rotated = rot.dot(points.T).T.astype(int)  # converted to int for bounding box working properly
#
#     # first filter find rotated segments that intersected by y = (y of rotated center of image)
#     # in other terms find segments intersected with line from drawOrient()
#     filter1 = []
#     for i in range(1, len(rotated), 2):
#         if (rotated[i][1] >= rotated[0][1] >= rotated[i + 1][1]) or \
#                 (rotated[i][1] <= rotated[0][1] <= rotated[i + 1][1]):
#             filter1.append(rotated[i])
#             filter1.append(rotated[i + 1])
#     filter1 = np.array(filter1)
#
#     # find max and min values of y in filter, to understand range of y of barcode
#     maxy = filter1[0][1]
#     miny = filter1[0][1]
#     for point in filter1:
#         if point[1] > maxy:
#             maxy = point[1]
#         if point[1] < miny:
#             miny = point[1]
#
#     # second filter includes all segments which have intersection with range of y of barcode
#     filter2 = []
#     for i in range(1, len(rotated), 2):
#         if (rotated[i][1] >= miny or rotated[i + 1][1] >= miny) and \
#                 (rotated[i][1] <= maxy or rotated[i + 1][1] <= maxy) and \
#                 abs(rotated[i][1]-rotated[i + 1][1]) > (maxy - miny) / 3:
#             filter2.append(rotated[i])
#             filter2.append(rotated[i + 1])
#     filter2 = np.array(filter2)
#
#     # build bounding rectangle
#     tx, ty, bw, bh = cv.boundingRect(filter2)
#     # add half of width and length so tx, ty are coordinates of barcode center
#     tx += bw / 2
#     ty += bh / 2
#
#     # reverse rotate to get center point
#     center = rotrev.dot(np.array([tx, ty]).T).T.astype(int)
#
#     whitespace = 10  # value to increase width to include whitespace
#     # get rotated rectangle
#     rot_rectangle = ((int(center[0]), int(center[1])), (bw + whitespace * 2, bh), angle/np.pi*180 - 90)
#
#     # draw rotated rectangle on source image
#     box = cv.boxPoints(rot_rectangle)
#     box = np.int0(box)
#     cv.drawContours(source, [box], 0, (0, 0, 255), 2)
#
#     # Show source
#     # cv.imshow("source", source)
#     # cv.waitKey(0)
#
#     # Show hisogram
#     # plt.xlim(0., np.pi)
#     # plt.show()
#
#     return source, (angle - np.pi/2)*180/np.pi, rot_rectangle


def drawOrient(angle, source):
    """
    draw red line perpendicular to barcode through center of image
    :param angle: angle of line
    :param source:
    """
    h, w = source.shape[:2]
    lineAngle = (angle + np.pi / 2)
    # print(lineAngle,np.cos(lineAngle),np.sin(lineAngle))
    r = 300  # half of length of red line
    pt1 = (round(w/2 + np.cos(lineAngle) * r),
           round(h/2 + np.sin(lineAngle) * r))

    pt2 = (round(w/2 - np.cos(lineAngle) * r),
           round(h/2 - np.sin(lineAngle) * r))
    cv.line(source, pt1, pt2, (0, 0, 255), 2)


# def rotate_image(mat, angle, color=(0, 0, 0)):
#
#     height, width = mat.shape[:2] # image shape has 3 dimensions
#     image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
#
#     rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)
#
#     # rotation calculates the cos and sin, taking absolutes of those.
#     abs_cos = abs(rotation_mat[0,0])
#     abs_sin = abs(rotation_mat[0,1])
#
#     # find the new width and height bounds
#     bound_w = int(height * abs_sin + width * abs_cos)
#     bound_h = int(height * abs_cos + width * abs_sin)
#
#     # subtract old image center (bringing image back to origo) and adding the new image center coordinates
#     rotation_mat[0, 2] += bound_w/2 - image_center[0]
#     rotation_mat[1, 2] += bound_h/2 - image_center[1]
#
#     # rotate image with the new bounds and translated rotation matrix
#     rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=color, interpolation=cv.INTER_CUBIC)
#     return rotated_mat


def crop_rect(img, rect):
    """
    Crop rotated rectangle from image in x4 dimensions of dimensions of rectangle to reduce data loss
    :param img:
    :param rect:
    :return: cropped image
    """
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    box = np.int0(rect[0])
    src_pts = rect[0].astype("float32")

    # width = int(rect[1][0])
    # height = int(rect[1][1])
    height = math.ceil(np.linalg.norm(rect[0][0]-rect[0][1]))
    width = math.ceil(np.linalg.norm(rect[0][1]-rect[0][2]))

    k = 4  # scale up coefficient
    # print(height, width)

    # dst_pts = np.array([[0, height * k - 1],
    #                     [0, 0],
    #                     [width * k - 1, 0],
    #                     [width * k - 1, height * k - 1]], dtype="float32")
    dst_pts = np.array([[0, 400 - 1],
                        [0, 0],
                        [1000 - 1, 0],
                        [1000 - 1, 400 - 1]], dtype="float32")
    # print(src_pts, dst_pts)
    # print('asd')
    # build transformation matrix, and perform tramsformation
    # Perspective can be changed to Affine, but require different input
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    # warped = cv.warpPerspective(img, M, (width * k, height * k), borderMode=cv.BORDER_REPLICATE, flags=cv.INTER_CUBIC)
    warped = cv.warpPerspective(img, M, (1000, 400), borderMode=cv.BORDER_REPLICATE, flags=cv.INTER_CUBIC)

    return warped


# def test(path):
#     """
#     Get image, find angle and perform cropping of rotated rectangle
#     :param path:
#     :return: (
#         image with lines drawed on,
#         cropped image
#     )
#     """
#     gray, source = getGrayImg(path)
#     sourcecopy = source.copy()
#     res, angle, rect = findLines(gray, source)
#     crop = crop_rect(sourcecopy, rect)
#     # cv.imshow("crop", crop)
#     # cv.waitKey(0)
#     return res, crop


# def checkFolder(dir):
#     """
#     Do test() on all files in directory and saves result images to nearby directories
#     :param dir: directory
#     """
#     samples = listdir(dir)
#     dest = dir + 'Detect'  # directory for images with lines
#     destR = dir + 'Rotated'  # directory for rotated cropped images
#     for pic in samples:
#         pathdestR = '\\'.join([destR, pic])
#         pathdest = '\\'.join([dest, pic])
#         path = '\\'.join([dir, pic])
#         print('Begin {}'.format(path))
#         res, rotated_img = test(path)
#         cv.imwrite(pathdest, res)
#         cv.imwrite(pathdestR, rotated_img)


if __name__ == '__main__':
    # test('CrpIMG_20220524_170606.jpg')
    # test('CrpIMG_20220524_170723.jpg')
    # test('CrpIMG_20220524_170825.jpg')
    # test('CrpIMG_20220524_170909.jpg')
    # test('CrpIMG_20220524_180149.jpg')

    # checkFolder('sample')  # test for 120crop set
    # checkFolder('100cm')
    pass

