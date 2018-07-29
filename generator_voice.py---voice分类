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

