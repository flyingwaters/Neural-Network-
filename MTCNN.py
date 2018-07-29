###detect.py
import os
import wave
import gc
import numpy as np
from pydub import AudioSegment
import tensorflow as tf
from sklearn.model_selection import train_test_split
"""
w = tf.Variable([1,2,3],[3,4,5],dtype=tf.float32,name="weights")
"""
train = []
label = []
test = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
i = -1
filepath = "D:/FFOutput/"
for filedir in os.listdir(filepath):
    f = os.path.join(filepath,filedir)
    i+=1
    for filename in os.listdir(f):
        final_filename = os.path.join(f,filename)
        with wave.open(final_filename) as file_object:
            params = file_object.getparams()
            nchannels,sampwidth,framerate,nframes = params[:4]

            strData = file_object.readframes(nframes)
            #读取音频，字符串格式

            waveData = np.fromstring(strData,dtype=np.int16)
            #转为数字
            waveData = waveData*1.0/(max(abs(waveData)))
            w_size = len(waveData)
            waveData=np.pad(waveData,(0,260000-w_size),mode='constant',constant_values=(0,0))

            #补充‘0’成为 一个 260000长度的 numpy数组

            #wave幅值归一化
            #vector for voice
            train.append(waveData)
            #train 数据
            label.append(test[i])
            #label 数据
print(label)
x_train,x_test,y_train,y_test = train_test_split(train,label,test_size=0.1,random_state=0)

batch_id = 0

def next(batch_size):
    global batch_id
    ###清除函数占用的内存
    for x in locals().keys():
        del locals()[x]
    gc.collect()
    # 当前函数生成的开销都给清空掉即可释放内存
    if batch_id == len(x_train):
        batch_id = 0
    if batch_id+batch_size > len(x_train):
        batch_id = len(x_train)-batch_id+batch_size
    batch_data = (x_train[batch_id:(batch_id+batch_size)])
    batch_labels = (y_train[batch_id:(batch_id+batch_size)])
    batch_id = (batch_id + batch_size)%len(x_train)
    return list(batch_data), list(batch_labels)
    ##############################
    ######gen_cebel_data.py
import os
from PIL import Image
import numpy as np
from tool import utils
import traceback

anno_src = r"D:\360\CelebA\Anno\list_bbox_celeba.txt"
img_dir = r"D:\360\CelebA\Img\img_celeba.7z\img_celeba"

save_path = r"D:\celeba4"

for face_size in [24,48]:

    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")
    print("gen %i image" % face_size)

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 样本描述存储路径
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue
            try:
                strs = line.strip().split(" ")
                strs = list(filter(bool, strs))
                image_filename = strs[0].strip()
                print(image_filename)
                image_file = os.path.join(img_dir, image_filename)

                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    px1 = 0#float(strs[5].strip())
                    py1 = 0#float(strs[6].strip())
                    px2 = 0#float(strs[7].strip())
                    py2 = 0#float(strs[8].strip())
                    px3 = 0#float(strs[9].strip())
                    py3 = 0#float(strs[10].strip())
                    px4 = 0#float(strs[11].strip())
                    py4 = 0#float(strs[12].strip())
                    px5 = 0#float(strs[13].strip())
                    py5 = 0#float(strs[14].strip())

                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    boxes = [[x1, y1, x2, y2]]

                    # 计算出人脸中心点位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    # 使正样本和部分样本数量翻倍
                    for _ in range(5):
                        # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(-0.2*w, 0.2*w)
                        h_ = np.random.randint(-0.2*h, 0.2*h)
                        cx_ = cx + w_
                        cy_ = cy + h_

                        # 让人脸形成正方形，并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                        x1_ = np.max(cx_ - side_len / 2, 0)
                        y1_ = np.max(cy_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])

                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        offset_px1 = 0#(px1 - x1_) / side_len
                        offset_py1 = 0#(py1 - y1_) / side_len
                        offset_px2 = 0#(px2 - x1_) / side_len
                        offset_py2 = 0#(py2 - y1_) / side_len
                        offset_px3 = 0#(px3 - x1_) / side_len
                        offset_py3 = 0#(py3 - y1_) / side_len
                        offset_px4 = 0#(px4 - x1_) / side_len
                        offset_py4 = 0#(py4 - y1_) / side_len
                        offset_px5 = 0#(px5 - x1_) / side_len
                        offset_py5 = 0#(py5 - y1_) / side_len

                        # 剪切下图片，并进行大小缩放
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        if iou > 0.65:  # 正样本
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    positive_count, 1, offset_x1, offset_y1,
                                    offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            positive_anno_file.flush()
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1
                        elif iou > 0.4:  # 部分样本
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    part_count, 2, offset_x1, offset_y1,offset_x2,
                                    offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.3:
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

                        # 生成负样本
                        _boxes = np.array(boxes)

                    for i in range(5):
                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if np.max(utils.iou(crop_box, _boxes)) < 0.3:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                            negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1
            except Exception as e:
                traceback.print_exc()


    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()

###########################################3
#net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3),  # conv1
            nn.PReLU(),  # PReLU1
            nn.MaxPool2d(kernel_size=2,stride=2),  # max pooling
            nn.Conv2d(10, 16, kernel_size=3.),  # conv2
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3),  # conv3
            nn.PReLU()  # PReLU3
        )
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)
        self.conv4_3 = nn.Conv2d(32, 10,kernel_size=1)
    def forward(self, x):
        x = self.pre_layer(x)
        cond = F.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cond, offset

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            nn.PReLU()  # prelu3

        )
        self.conv4 = nn.Linear(64 * 2 * 2, 128)  # conv4
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        self.conv5_3 = nn.Linear(128, 10)


    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        label = F.sigmoid(self.conv5_1(x))
        offset = self.conv5_2(x)
        return label, offset


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1  48
            nn.PReLU(),  # prelu1 46
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1  23
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2 21
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2 10
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3 8
            nn.PReLU(),  # prelu3
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3 4
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
            nn.PReLU()  # prelu4
        )
        self.conv5 = nn.Linear(128 * 2 * 2, 256)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)
    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        label = F.sigmoid(self.conv6_1(x))
        offset = self.conv6_2(x)
        return label, offset
        
        #######################
####dataset.py        
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image

#重写dataset 重写两个函数的__init__()  ,  __getitem__()

class FaceDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())
    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        img_path = os.path.join(self.path, strs[0])
        cond = torch.Tensor([int(strs[1])])
        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
        img_data = Image.open(img_path)
        #one_line code is easy to be
        img_data = np.array(img_data,dtype=np.float32)
        img_data = np.transpose(img_data,(2,0,1))
        img_data = torch.Tensor(img_data)
        return img_data, cond, offset
    def __len__(self):
        return len(self.dataset)
####
#############################################
######trainer.py
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from simpling import FaceDataset
class Trainer:
    def __init__(self, net, save_path, dataset_path, isTrainLandmark=False, isCuda=True):
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda
        self.isTrainLandmark = isTrainLandmark
        if self.isCuda:
            self.net.cuda()
        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()
        # if isTrainLandmark:
        #     self.landmark_loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())
        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
    def train(self):
        faceDataset = FaceDataset(self.dataset_path)
        dataloader = DataLoader(faceDataset, batch_size=128, shuffle=True, num_workers=4)
        o= 0
        while o<200:
            o+=1
            for i, (_img_data, _category, _offset) in enumerate(dataloader):
                if self.isCuda:
                    _img_data = _img_data.cuda()
                    category_ = _category.cuda()
                    offset_ = _offset.cuda()
                    # if self.isTrainLandmark:
                    #     landmark_ = landmark_.cuda()
                _output_category, _output_offset = self.net(_img_data)
                # 计算分类的损失
                category_mask = torch.lt(category_, 2)
                # part样本不参与分类损失计算
                category = category_[category_mask]
                output_category = _output_category[category_mask]
                cls_loss = self.cls_loss_fn(output_category, category)
                #  两个目标：一个输出 VS 一个输入
                #交叉熵
                # 计算bound的损失
                offset_mask = torch.gt(category_, 0)  # 负样本不参与计算
                mask = torch.cat((offset_mask,offset_mask),1)
                mask2 = torch.cat((mask,mask),1)
                offset = offset_[mask2]
                offset = offset.view(-1,4)
                output_offset = _output_offset[mask2]
                output_offset=output_offset.view(-1,4)
                offset_loss = self.offset_loss_fn(output_offset, offset)  # 损失
                # 计算landmark的损失
                # if self.isTrainLandmark:
                #     landmark_mask = torch.gt(category_, 0)  # 负样本不参与计算
                #     landmark_index = torch.nonzero(landmark_mask)[:, 0]  # 选出非负样本的索引
                #     landmark = landmark_[landmark_index]
                #     output_landmark = _output_landmark[landmark_index]
                #     landmark_loss = self.landmark_loss_fn(output_landmark, landmark)  # 损失
                #
                #     loss = cls_loss + offset_loss + landmark_loss  # 总loss
                # else:
                loss = cls_loss + offset_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # # if self.isTrainLandmark:
                # #     print(" loss:", loss.cpu().data.numpy(), " cls_loss:", cls_loss.cpu().data.numpy(), " offset_loss",
                # #           offset_loss.cpu().data.numpy(), " landmark_loss",
                # #           landmark_loss.cpu().data.numpy())
                # else:
                print("{0} loss:".format(o),loss.cpu().data.numpy(), " cls_loss:", cls_loss.cpu().data.numpy(), " offset_loss",
                    offset_loss.cpu().data.numpy())
            torch.save(self.net.state_dict(), self.save_path)
            print("save success")
 ###############3
 ###########
 ######三个网络得训练脚本
 import nets
import train
if __name__ == '__main__':
    print("start")
    net = nets.ONet()
    trainer = train.Trainer(net, 'D:/param/onet.pt', r"D:/celeba4/48")
    trainer.train()
    print("end up")
    
    import nets
import train
if __name__ == '__main__':
    net = nets.PNet()
    trainer = train.Trainer(net, r'D:/param/pnet.pt', r"D:\celeba4\12")
    print("..........")
    trainer.train()
    print("..........")
   
 import nets
import train
if __name__ == '__main__':
    net = nets.RNet()
    print("————")
    trainer = train.Trainer(net, 'D:/param/rnet.pt', r"D:\celeba4\24")
    trainer.train()
    print("————")
