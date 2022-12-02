# import the necessary packages
import torch.nn.functional as F
import torch
from multiprocessing import Pool
import numpy as np
import time
import cv2
import os
# dlib部分都删掉


#将文件名按数字顺序排序
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


#删除了“size匹配”这一块的内容, 因为这样会导致每个人脸大小不同
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


#在宽和高两个方向计算人脸在两个方向上的变化
def calculate(idx, all_frames, save_path_w, save_path_h):
    global dirpath, name_w, name_h, num_w, num_h
    t0 = time.time()
    directory_name = dirpath.split(os.sep)
    directory_name = directory_name.pop(-2)
    prev_avg = None
    # RGB2YUV
    a = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
    b = np.array([0, 128, 128])
    all_frames = all_frames.dot(a) + b
    # 再新建一个 宽作为图片从左到右 new_frame =（第一维度，第二维度，第四维度)
    for index in range(0, all_frames.shape[2]):
        # 在每个宽度上取矩阵new_frames
        new_frame = all_frames[:, :, index, :]
        time_axis = new_frame.shape[0]
        height = new_frame.shape[1]
        grid_h = 8
        grid_t = 4
        kernel = (time_axis // grid_t, height // grid_h)
        # 取平均值
        avg_frame = F.avg_pool2d(torch.FloatTensor(np.transpose(new_frame, (2, 0, 1))), kernel_size=kernel,
                                 stride=kernel)
        if prev_avg is not None:
            if tuple(avg_frame.shape) != (3, grid_t, grid_h):  # kernel小于 grid的时候会产生多余网格
                avg_frame = avg_frame[:, :grid_t, :grid_h]
            prev_avg = np.concatenate((prev_avg, avg_frame.reshape((3, grid_t * grid_h, 1))), axis=2)
        else:
            prev_avg = avg_frame.reshape((3, grid_t * grid_h, 1))
        if prev_avg.shape[2] == all_frames.shape[2]:
            head = ''
            pwd = dirpath
            pwd = pwd.split(os.sep)
            del pwd[-2]
            if pwd[-1] == '':
                del pwd[-1]
            for item in pwd:
                head = head + item + os.sep
            head = os.path.join(save_path_w, os.path.relpath(head, root)) + os.sep
            np.save('%s%s.npy' % (head, directory_name), np.transpose(prev_avg, (1, 2, 0)))
            cv2.imwrite('%s%s.jpg' % (head, directory_name), np.transpose(prev_avg, (1, 2, 0)).astype(np.uint8))
            cv2.destroyAllWindows()
            prev_avg = None
            name_w += 1
        num_w += 1
    print('Width direction. video#%d video:%s running time:%fs' % (idx, directory_name, time.time() - t0))

    # 在新建第二个 高作为图片从左到右 new_frame = (第一维度，第三维度，第四维度)
    t0 = time.time()
    for index in range(0, all_frames.shape[1]):
        new_frame = all_frames[:, index, :, :]
        time_axis = new_frame.shape[0]
        width = new_frame.shape[1]
        grid_w = 8
        grid_t = 4
        kernel = (time_axis // grid_t, width // grid_w)
        #kernel_size是不是先宽再高
        avg_frame = F.avg_pool2d(torch.FloatTensor(np.transpose(new_frame, (2, 0, 1))), kernel_size=kernel,
                                 stride=kernel)
        if prev_avg is not None:
            if tuple(avg_frame.shape) != (3, grid_t, grid_w):  # kernel小于 grid的时候会产生多余网格
                avg_frame = avg_frame[:, :grid_t, :grid_w]
            prev_avg = np.concatenate((prev_avg, avg_frame.reshape((3, grid_t * grid_w, 1))), axis=2)
        else:
            prev_avg = avg_frame.reshape((3, grid_t * grid_w, 1))
        if prev_avg.shape[2] == all_frames.shape[1]:
            head = ''
            pwd = dirpath
            pwd = pwd.split(os.sep)
            del pwd[-2]
            if pwd[-1] == '':
                del pwd[-1]
            for item in pwd:
                head = head + item + os.sep
            head = os.path.join(save_path_h, os.path.relpath(head, root)) + os.sep
            np.save('%s%s.npy' % (head, directory_name), np.transpose(prev_avg, (1, 2, 0)))
            cv2.imwrite('%s%s.jpg' % (head, directory_name), np.transpose(prev_avg, (1, 2, 0)).astype(np.uint8))
            cv2.destroyAllWindows()
            prev_avg = None
            name_h += 1
        num_h += 1
    print('Height direction. video#%d video:%s running time:%fs' % (idx, directory_name, time.time() - t0))


#主函数
def main(idx, save_path_t):
    global prev_template, file_list, dirpath
    # directory_name是视频文件夹的名字, dirpath是输入视频文件夹的路径, file_list是输入视频文件的所有帧的列表
    directory_name = dirpath.split(os.sep)
    directory_name = directory_name.pop(-2)
    # if num == 0:
    file = file_list[0]
    t0 = time.time()
    num = 0
    name = 1
    break_if_one2one = False
    prev_avg = prev_template = None
    frame = cv2.imread(os.path.join(dirpath, file), flags=cv2.IMREAD_COLOR)
    while os.path.isfile(os.path.join(dirpath, file)):
        if not face_root:
            # unaligned // 4, aligned / 5.333 但是aligned数据暂时还不能用
            x1, y1, x2, y2 = (frame.shape[1] // 4, frame.shape[0] // 4, 3*frame.shape[1] // 4, 3*frame.shape[0] // 4)
            # x1, y1, x2, y2 = (int(frame.shape[1] / 5.333), int(frame.shape[0] / 5.333), int(4.333 * frame.shape[1] / 5.333), int(4.333 * frame.shape[0] / 5.333))
            noface = True if x1 >= x2 or y1 >= y2 else False
        if not noface:
            xx1, yy1, xx2, yy2 = face_location_match(np.copy(frame), x1, y1, x2, y2)

            # crop face
            face_im = frame[max(0, yy1):min(yy2, frame.shape[0]), max(0, xx1):min(xx2, frame.shape[1])]

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
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # cv2.rectangle(frame, (xx1, yy1), (xx2, yy2), (0, 255, 0), 1)
            # cv2.circle(frame, center, 3, (0, 255, 0), 1)
            # frame = cv2.resize(frame, (540, int(540*frame.shape[0]/frame.shape[1])))
            if prev_avg.shape[2] == grid_w*grid_h and not break_if_one2one:
                # 计算输出路径。这里dirpath是后面有一个斜杠的
                head = ''
                pwd = dirpath
                pwd = pwd.split(os.sep)
                del pwd[-2]
                if pwd[-1] == '':
                    del pwd[-1]
                for item in pwd:
                    head = head + item + os.sep
                head = os.path.join(save_path_t, os.path.relpath(head, root)) + os.sep
                np.save('%s%s.npy' % (head, directory_name), np.transpose(prev_avg, (1, 2, 0)))
                cv2.imwrite('%s%s.jpg' % (head, directory_name), np.transpose(prev_avg, (1, 2, 0)).astype(np.uint8))
                cv2.destroyAllWindows()
                if one2one:
                    break_if_one2one = True
                prev_avg = None
                name += 1
            num += 1
        if num == 1:
            all_frames = face_im[np.newaxis, :]
        else:
            all_frames = np.concatenate((all_frames, face_im[np.newaxis, :]), axis=0)
        if num < len(file_list):
            file = file_list[num]
        else:
            file = 'Done'
        # 读取下一张图（一帧）
        frame = cv2.imread(os.path.join(dirpath, file), flags=cv2.IMREAD_COLOR)
        if num % 8 == 0:
            print('Time direction. video#%d file:%s video:%s running time:%fs' % (idx, file, directory_name, time.time()-t0))
    return idx, all_frames


#搜索输入文件夹的每个图片并且计算和输出图像
def search(path, save_path_one, save_path_two, save_path_three):
    global dirpath, file_list, y
    dirs = []
    files = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            dirs.append(item)
        else:
            files.append(item)
    if dirs:
        for folder in dirs:
            new_path = os.path.join(path, folder)
            search(new_path, save_path_one, save_path_two, save_path_three)
    else:
        if files:
            dirpath = path + os.sep
            file_list = sort_file(files)
            # 从第一张图开始运行main。
            y, frames = main(y, save_path_one)
            calculate(y, frames, save_path_two, save_path_three)
            y += 1
        else:
            print("No data file found.")
            return


#根据数据文件夹层次结构新建一个相同的空文件夹结构到保存的地方并且删掉最深层的文件夹（这层文件夹是数据中包含每一个视频文件的文件夹）
# def create_folders(path, save_path):
#     dirs = []
#     files = []
#     for item in os.listdir(path):
#         if os.path.isdir(os.path.join(path, item)):
#             dirs.append(item)
#         else:
#             files.append(item)
#     if dirs:
#         for folder in dirs:
#             if folder == dirs[0]:
#                 save_path = os.path.join(save_path, folder)
#             else:
#                 save_path = os.path.join(os.path.dirname(save_path), folder)
#             if not os.path.exists(save_path):
#                 os.mkdir(save_path)
#             new_path = os.path.join(path, folder)
#             create_folders(new_path, save_path)
#             if create_folders(new_path, save_path) == 0:
#                 os.rmdir(save_path)
#     else:
#         return 0

#思路还未实现
def create_folders(path, save_path):
    global dirpath
    #先去掉最后一个元素然后直接用makedirs就行了
    for dir in os.listdir(path):
        path1 = os.path.join(save_path, dir)
        if not os.path.exists(path1):
            os.makedirs(path1)

center = size = prev_face = prev_template = None
face_root = None
# 是否只输出一个图像
one2one = True
source_data = 'oulu_image'
if source_data == 'oulu':
    save_root = 'STmap/oulu32/'
    root = '/home/yaowen/Documents/1Database/PAD-datasets/Oulu-NPU'
    face_root = '/home/yaowen/Documents/1Database/PAD-datasets/Oulu-NPU/face-location-S3FD'
if source_data == 'SiW':
    save_root = 'STmap/SiW32/'
    root = face_root = '/home/yaowen/Documents/1Database/PAD-datasets/SiW/SiW_release'
elif source_data == 'CASIA':
    save_root =  'STmap/CASIA32/'
    root = '/home/yaowen/Documents/1Database/PAD-datasets/CASIA-FASD'
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
    root = '\\Users\\ziywa\\PycharmProjects\\pythonProject\\samples\\oulu_image\\unaligned\\'
    save_root = 'STmap\\mysamples\\oulu_image\\'
# 查看source_data是不是上面其中之一
try:
    'save_root' in locals().keys() and 'root' in locals().keys()
except Exception as ex:
    print("Error: Either save root or root does not exist")
#没有save_root则创建
if not os.path.exists(save_root):
    os.makedirs(save_root)
num_workers = 1
# result = []
# 初始化全局变量
dirpath = ''
file_list = []
y = 1
name_w = 1
name_h = 1
num_w = 0
num_h = 0
frames = np.array([0, 0, 0])
# 创建时间宽高三个文件夹
path_t = os.path.join(save_root, 'Time' + os.sep)
if not os.path.exists(path_t):
    os.mkdir(path_t)
path_w = os.path.join(save_root, 'Width' + os.sep)
if not os.path.exists(path_w):
    os.mkdir(path_w)
path_h = os.path.join(save_root, 'Height' + os.sep)
if not os.path.exists(path_h):
    os.mkdir(path_h)
# 在三个文件夹里面根据训练数据文件夹的结构和名字创建一样的文件夹并删掉最深层的
create_folders(root, path_t)
create_folders(root, path_w)
create_folders(root, path_h)
# 如果 num_workers > 1 是多进程，现在还不能正常使用
#if num_workers > 1:
#    if __name__ == '__main__':
#        pool = Pool(num_workers)
#        pool.apply_async(search, args=(root, path1, path2, path3))
#else:
search(root, path_t, path_w, path_h)
