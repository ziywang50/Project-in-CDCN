# import the necessary packages
import torch.nn.functional as F
import torch
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from multiprocessing import Pool
import numpy as np
import imutils
import time
#import dlib
import cv2
import os

# initialize dlib's face detector (Histogram of Oriented Gradients HOG-based 方向梯度直方图+线性分类器)
# and then create the facial landmark predictor (Ensemble of Regression Trees)
print("[INFO] loading facial landmark predictor...")
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks
right_eye = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
left_eye = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]


def sort_file(file_list):
    int_list = []
    for name in file_list:
        int_list.append(int(name.split('.jpg')[0]))
    int_list.sort()
    sort_list = []
    for sort_num in int_list:
        for file in file_list:
            if str(sort_num) == file.split('.jpg')[0]:
                sort_list.append(file)
    return sort_list


def list_path(root, recursive, exts):
    i = 0
    if recursive:
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files = sort_file(files)
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    yield (i, os.path.relpath(fpath, root))
                    i += 1
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1


def face_location(face_txt_path):
    if not os.path.exists(face_txt_path):
        print('no %s' % face_txt_path)
        return False
    f = open(face_txt_path)
    faces = []
    for line in f:
        xy = [int(x) for x in line.strip().split()]
        faces.append(xy)
        # faces.append([int(x) for x in line.strip().split(', ')[1:5]]) # for MSU
    f.close()
    return faces


def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def diff(prev_frame, frame):
    res = frame.astype(np.int16) - prev_frame.astype(np.int16)
    mean = abs(res).mean()
    res = res*4+128
    np.where(res<0,0,res)
    np.where(res>255,255,res)
    return res.astype(np.uint8), mean


def face_location_match(frame, x1, y1, x2, y2):
    global center, size, prev_face, prev_template
    if prev_template is not None:
        # 中心点匹配
        match_result = cv2.matchTemplate(frame[max(0, y1):min(y2, frame.shape[0]), max(0, x1):min(x2, frame.shape[1])],
                                   prev_template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
        tl = max_loc
        center = (tl[0] + size[0] // 8 + max(0, x1), tl[1] + size[1] // 8 + max(0, y1))  # 匹配到的左上角+模板半长宽+目标左上角坐标
        # # size匹配
        # face_diff_dict = {}
        # for i in range(-10, 11, 2):
        #     sizex = size[0] + i
        #     sizey = int(sizex * size[1] / size[0])
        #     xx1 = center[0] - sizex//2
        #     xx2 = center[0] + sizex//2
        #     yy1 = center[1] - sizey//2
        #     yy2 = center[1] + sizey//2
        #     # crop
        #     res_img = frame[max(0, yy1):min(yy2, frame.shape[0]), max(0, xx1):min(xx2, frame.shape[1])]
        #     res_img = cv2.resize(res_img, (prev_face.shape[1], prev_face.shape[0]))
        #     # diff
        #     res_diff, mean = diff(prev_face, res_img)
        #     face_diff_dict[mean] = [res_img, res_diff, (sizex, sizey)]
        # minres = min(face_diff_dict)
        # size = face_diff_dict[minres][2]
        # cv2.imshow("diff", face_diff_dict[minres][1])
        # cv2.waitKey(100)
    else:
        center = ((x2 + x1) // 2, (y2 + y1) // 2)  # 第一帧以检测为准
        size = ((x2 - x1) // 2 * 2, (y2 - y1) // 2 * 2)
    xx1 = center[0] - size[0] // 8
    xx2 = center[0] + size[0] // 8
    yy1 = center[1] - size[1] // 8
    yy2 = center[1] + size[1] // 8
    prev_template = frame[yy1:yy2, xx1:xx2]
    xx1 = center[0] - size[0] // 2
    xx2 = center[0] + size[0] // 2
    yy1 = center[1] - size[1] // 2
    yy2 = center[1] + size[1] // 2
    prev_face = frame[max(0, yy1):min(yy2, frame.shape[0]), max(0, xx1):min(xx2, frame.shape[1])]
    return xx1, yy1, xx2, yy2


def remove_bg(img, xys):
    hull = cv2.convexHull(xys)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [hull], -1, (1, 1, 1), -1)
    return img * mask


def calculate(all_frames):
    global num, prev_avg, name
    num = 0
    prev_avg = None
    name = 1
    # RGB2YUV
    a = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
    b = np.array([0, 128, 128])
    all_frames = all_frames.dot(a) + b
    # 再新建一个（第一维度，第二维度，第四维度), 宽作为图片从左到右
    for index in range(0, all_frames.shape[2]):
        new_frame = all_frames[:, :, index, :]
        # 取平均值
        time_axis = new_frame.shape[0]
        height = new_frame.shape[1]
        grid_h = 8
        grid_t = 4
        kernel = (time_axis // grid_t, height // grid_h)
        avg_frame = F.avg_pool2d(torch.FloatTensor(np.transpose(new_frame, (2, 0, 1))), kernel_size=kernel,
                                 stride=kernel)
        if prev_avg is not None:
            if tuple(avg_frame.shape) != (3, grid_t, grid_h):  # kernel小于 grid的时候会产生多余网格
                avg_frame = avg_frame[:, :grid_t, :grid_h]
            prev_avg = np.concatenate((prev_avg, avg_frame.reshape((3, grid_t * grid_h, 1))), axis=2)
        else:
            # print(avg_frame.shape)
            # change avg_frame shape to smaller
            prev_avg = avg_frame.reshape((3, grid_t * grid_h, 1))
        if prev_avg.shape[2] == all_frames.shape[2]:
            # video_writer.release()
            var_mean = prev_avg.var(axis=2).var()
            print("var_mean = ", var_mean)
            # with open('matched_video_var.txt', 'a') as f:
            #     f.write(file + str(name) + '\t' + str(var_mean) + '\n')
            np.save('%s%s_%s.npy' % (save_root, source_data, name if name != 0 else '1'),
                    np.transpose(prev_avg, (1, 2, 0)))
            cv2.imwrite('%s%s_%s.jpg' % (save_root, source_data, name if name != 0 else '1'),
                        np.transpose(prev_avg, (1, 2, 0)).astype(np.uint8))
            cv2.destroyAllWindows()
            if one2one:
                break
            prev_avg = None
            name += 1
            # video_writer = cv2.VideoWriter('%s%s%s.avi' % (save_root, file.split('.')[0], newname),
            #                               cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            #                               8, (540, int(540 * vs.get(4) / vs.get(3))))
        num += 1

    # 在新建第二个（第一维度，第三维度，第四维度），高作为图片从左到右
    for index in range(0, all_frames.shape[1]):
        new_frame = all_frames[:, index, :, :]
        time_axis = new_frame.shape[0]
        width = new_frame.shape[1]
        grid_w = 8
        grid_t = 4
        kernel = (time_axis // grid_t, width // grid_w)
        avg_frame = F.avg_pool2d(torch.FloatTensor(np.transpose(new_frame, (2, 0, 1))), kernel_size=kernel,
                                 stride=kernel)
        if prev_avg is not None:
            if tuple(avg_frame.shape) != (3, grid_t, grid_w):  # kernel小于 grid的时候会产生多余网格
                avg_frame = avg_frame[:, :grid_t, :grid_w]
            prev_avg = np.concatenate((prev_avg, avg_frame.reshape((3, grid_t * grid_w, 1))), axis=2)
        else:
            # print(avg_frame.shape)
            # change avg_frame shape to smaller
            prev_avg = avg_frame.reshape((3, grid_t * grid_w, 1))
        if prev_avg.shape[2] == all_frames.shape[1]:
            # video_writer.release()
            var_mean = prev_avg.var(axis=2).var()
            print("var_mean = ", var_mean)
            # with open('matched_video_var.txt', 'a') as f:
            #    f.write(file + str(name) + '\t' + str(var_mean) + '\n')
            np.save('%s%s_%s.npy' % (save_root, source_data, name if name != 0 else '1'),
                    np.transpose(prev_avg, (1, 2, 0)))
            cv2.imwrite('%s%s_%s.jpg' % (save_root, source_data, name if name != 0 else '1'),
                        np.transpose(prev_avg, (1, 2, 0)).astype(np.uint8))
            cv2.destroyAllWindows()
            if one2one:
                break
            prev_avg = None
            name += 1
            # video_writer = cv2.VideoWriter('%s%s%s.avi' % (save_root, file.split('.')[0], newname),
            #                               cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            #                               8, (540, int(540 * vs.get(4) / vs.get(3))))
        num += 1


def main(idx, file):
    global prev_template, all_frames, name, num, prev_avg, t0
    if num == 0:
        t0 = time.time()
        num = 0
        name = 1
        prev_avg = prev_template = None
    # 有数据了则跳过
    # if os.path.exists(save_root + file.split('.')[0] + '.npy'):
    #     print(save_root + file.split('.')[0] + '.npy done!')
    #     return
    # elif not os.path.exists(save_root + os.path.dirname(file)):
    #     os.makedirs(save_root + os.path.dirname(file))
    # 加载人脸位置
    if face_root is not None:
        faces = face_location(os.path.join(face_root, file.split('.')[0]) + '.face')
        if not faces:
            print('%s no face location' % file)
            return
    # 读取视频帧，检测landmarks
    # vs = cv2.VideoCapture(os.path.join(root, file))
    # ret, frame = vs.read()
    frame = cv2.imread(os.path.join(root, file), flags=cv2.IMREAD_COLOR)
    # video_writer = cv2.VideoWriter(save_root + file.split('.')[0]+'.avi',
    #                              cv2.VideoWriter_fourcc('M','J','P','G'),
    #                              8, (540, int(540*vs.get(4)/vs.get(3)))
    #                              )
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face detection
    if os.path.isfile(os.path.join(root, file)):
        if face_root:
            if faces[num][0] == 0 and len(faces[num]) > 4:
                frame = rotate(frame, 180)
            x1, y1, x2, y2 = faces[num][-4:]
        # rect = dlib.rectangle(x1, y1, x2, y2+3)
            noface = True if x1 >= x2 or y1 >= y2 else False
        # else:
            # rect = detector(gray, 0)
            # noface = True if len(rect) == 0 else False
        if not face_root:
            # aligned
            x1, y1, x2, y2 = (frame.shape[1]//4, frame.shape[0]//4, 3*frame.shape[1]//4, 3*frame.shape[0]//4)
            # ??
            noface = True if x1 >= x2 or y1 >= y2 else False
        if not noface:
            # landmarks
            # shape = predictor(gray, rect)
            # shape = face_utils.shape_to_np(shape)
            # start = shape.min(axis=0)
            # end = shape.max(axis=0)
            # start[1] = start[1] - int((end[1] - start[1]) * 0.2)

            #取形状和框架坐标的最小值
            #if start[0]<x1 or start[1] < y1 or end[0] > x2 or end[1] > y2+3:
            #    start[0] = rect.left()
            #    start[1] = rect.top()
            #    end[0] = rect.right()
            #    end[1] = rect.bottom()  # 有一些landmarks检测不太准确
            # face match

            #不用dlib先跳一步，不算形状坐标
            # xx1, yy1, xx2, yy2 = face_location_match(np.copy(frame), start[0], start[1], end[0], end[1])
            s = size
            xx1, yy1, xx2, yy2 = face_location_match(np.copy(frame), x1, y1, x2, y2)
            if size != s:
                print(num, s, size)
            # remove background
            # (j, k) = face_utils.FACIAL_LANDMARKS_IDXS['jaw']
            # cnt = shape[j:k]
            # cnt = np.concatenate((cnt, start.reshape((1, 2)), np.array([[end[0], start[1]]])), axis=0)
            # frame = remove_bg(frame, cnt) # 有一些landmarks检测不太准确

            # crop face
            # face_im = frame[max(0, y1):min(y2, frame.shape[0]), max(0, x1):min(x2, frame.shape[1])]
            face_im = frame[max(0, yy1):min(yy2, frame.shape[0]), max(0, xx1):min(xx2, frame.shape[1])]
            # take the face rectangle and split

            # RGB2YUV
            a = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
            b = np.array([0, 128, 128])
            face_im = face_im.dot(a) + b

            # 求分块均值
            height = face_im.shape[0]
            width = face_im.shape[1]
            grid_w = 4
            grid_h = 8
            kernel = (height // grid_h, width // grid_w)
            avg_frame = F.avg_pool2d(torch.FloatTensor(np.transpose(face_im, (2, 0, 1))), kernel_size=kernel,
                                 stride=kernel)
            if prev_avg is not None:
                if tuple(avg_frame.shape) != (3, grid_h, grid_w):   # kernel小于 grid的时候会产生多余网格
                    avg_frame = avg_frame[:, :grid_h, :grid_w]
                prev_avg = np.concatenate((prev_avg, avg_frame.reshape((3, grid_w*grid_h, 1))), axis=2)
            else:
                prev_avg = avg_frame.reshape((3, grid_w*grid_h, 1))

            # show result
            # for s in shape:
            #     cv2.circle(frame, tuple(s), 3, (0, 0, 255), 1)
            # cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.rectangle(frame, (xx1, yy1), (xx2, yy2), (0, 255, 0), 1)
            cv2.circle(frame, center, 3, (0, 255, 0), 1)
            frame = cv2.resize(frame, (540, int(540*frame.shape[0]/frame.shape[1])))
            cv2.imshow('x', frame)
            cv2.waitKey(100)
            # video_writer.write(frame)
            if prev_avg.shape[2] == grid_w*grid_h:
                # video_writer.release()
                var_mean = prev_avg.var(axis=2).var()
                with open('matched_video_var.txt', 'a') as f:
                    f.write(file + str(name) + '\t' + str(var_mean) + '\n')
                np.save('%s%s_%s.npy' % (save_root, file.split('.')[0], name if name != 0 else '1'), np.transpose(prev_avg, (1, 2, 0)))
                cv2.imwrite('%s%s_%s.jpg' % (save_root, file.split('.')[0], name if name != 0 else '1'), np.transpose(prev_avg, (1, 2, 0)).astype(np.uint8))
                cv2.destroyAllWindows()
                # if one2one:
                #    break
                prev_avg = None
                name += 1
            # video_writer = cv2.VideoWriter('%s%s%s.avi' % (save_root, file.split('.')[0], name),
            #                               cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            #                               8, (540, int(540*vs.get(4)/vs.get(3))))
            num += 1
        if num == 1:
            all_frames = face_im[np.newaxis, :]
        else:
            all_frames = np.concatenate((all_frames, face_im[np.newaxis, :]), axis=0)
        # frame = cv2.imread(os.path.join(root, file), flags=cv2.IMREAD_COLOR)
        print('%d %s %fs' % (idx, file, time.time()-t0))
    return all_frames


center = size = prev_face = prev_template = None
face_root = None
one2one = False
source_data = 'oulu_image'
if source_data == 'oulu':
    save_root = 'STmap/oulu32/'
    # root = face_root = '/home/yaowen/Documents/PycharmProjects/PAD200707/samples/oulu'
    root = '/home/yaowen/Documents/1Database/PAD-datasets/Oulu-NPU'
    face_root = '/home/yaowen/Documents/1Database/PAD-datasets/Oulu-NPU/face-location-S3FD'
if source_data == 'SiW':
    save_root = 'STmap/SiW32/'
    root = face_root = '/home/yaowen/Documents/1Database/PAD-datasets/SiW/SiW_release'
elif source_data == 'CASIA':
    save_root =  'STmap/CASIA32/'
    root = '/home/yaowen/Documents/1Database/PAD-datasets/CASIA-FASD'
    # root = face_root = './samples/CASIA'
    face_root = '/home/yaowen/Documents/1Database/PAD-datasets/CASIA-FASD/face-locations-S3FD'
elif source_data == 'sample':
    save_root = 'STmap/mysamples/'
    root = face_root = './samples/'
elif source_data == 'msu':
    save_root = 'STmap/msu32/'
    root = '/home/yaowen/Documents/1Database/PAD-datasets/MSU-MFSD/scene01/'
    face_root = '/home/yaowen/Documents/1Database/PAD-datasets/MSU-MFSD/face-location-S3FD'
elif source_data == 'replay':
    save_root = 'STmap/replay32/'
    root = '/home/yaowen/Documents/1Database/PAD-datasets/ReplayAttack'
    face_root = '/home/yaowen/Documents/1Database/PAD-datasets/ReplayAttack/replayattack-face-locations-v2/face-locations-S3FD'
elif source_data == 'oulu_image':
    root = './samples/oulu_image/unaligned/1_1_01_1'
    save_root = 'STmap/mysamples/oulu_image/'
if not os.path.exists(save_root):
    os.makedirs(save_root)

num_workers = 1
# video_ext = ['.mp4', '.avi', '.mov']
image_ext = ['.jpg']
result = []
num = 0
if num_workers > 1:
    pool = Pool(num_workers)
    for x, fpath in list_path(root, True, image_ext):
        # if 'real' in fpath or 'photo' in fpath:
        # if '_1.' in fpath or '_2.' in fpath or '_3.' in fpath:
        result.append(pool.apply_async(main, args=(x, fpath)))
    for i, data in enumerate(result):
        result[i] = data.get()
else:
    for x, fpath in list_path(root, True, image_ext):
        # if '_1.' in fpath or '_2.' in fpath or '_3.' in fpath or 'real' in fpath:
        # result.append(main(x, fpath))
        all_frames = main(x, fpath)
    calculate(all_frames)

# main(0, 'oulu/1_1_01_1.avi')
# root = face_root ='/home/yaowen/Documents/1Database/PAD-datasets/SiW/SiW_release'
# # main('attack/attack_client001_laptop_SD_printed_photo_scene01.mov')