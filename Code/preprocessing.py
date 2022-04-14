import numpy as np
import struct
import os 
from keras.utils.np_utils import to_categorical

def load_data(path,mode):
    if mode == 'train':
        file_path = os.path.join(path,'train-images.idx3-ubyte')
        label_path = os.path.join(path,'train-labels.idx1-ubyte')
    else:
        file_path = os.path.join(path,'t10k-images.idx3-ubyte')
        label_path = os.path.join(path,'t10k-labels.idx1-ubyte')
        
    binfile = open(file_path, 'rb') 
    buffers = binfile.read()
    magic,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    
    images = images.reshape((len(images),28,28,1))
    
    binfile = open(label_path, 'rb')
    buffers = binfile.read()
    magic,num = struct.unpack_from('>II', buffers, 0) 
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    
    images = images/255
    labels = to_categorical(labels)
    
    return images,labels

def partition_public_dataset(train_images, train_labels,r=0.2):
    # random_index = np.random.permutation(len(train_images))
    # np.save('random_index.npy',random_index)
    random_index = np.load('random_index.npy')
    
    num = int(r*len(train_images))

    public_images = train_images[random_index[:num]]

    train_images = train_images[random_index[num:]]
    train_labels = train_labels[random_index[num:]]
    
    return train_images,train_labels,public_images

def local_data_partition(train_labels,alpha=0.5,local_size=600):
    range_length = local_size
    CLASS_NUM = train_labels.shape[1]

    category_dict = {}
    category_used_data = {}
    train_users_dict = {}

    for cid in range(CLASS_NUM):
        category_used_data[cid] = 0
        flag = np.where(train_labels.argmax(axis=-1) == cid)[0]
        perb = np.random.permutation(len(flag))
        flag = flag[perb]
        category_dict[cid] = flag

    user_num = int(np.ceil(len(train_labels)//range_length))
    for uid in range(user_num-1):
        p = np.random.dirichlet([alpha]*CLASS_NUM, size=1)*range_length
        p = np.array(np.round(p),dtype='int32')[0]
        ix = p.argmax()
        p[ix] = p[ix] - (p.sum()-range_length)
        assert p.sum() == range_length and (p>=0).mean() == 1.0

        data = []
        for cid in range(CLASS_NUM):
            s = category_used_data[cid]
            ed = s + p[cid]
            category_used_data[cid] += p[cid]

            data.append(category_dict[cid][s:ed])
        data = np.concatenate(data)
        train_users_dict[uid] = data

    data = []
    for cid in range(CLASS_NUM):
        s = category_used_data[cid]        
        data.append(category_dict[cid][s:])
    data = np.concatenate(data)
    train_users_dict[user_num-1] = data

    train_users = train_users_dict
    
    return train_users