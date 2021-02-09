import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.image as mpimg
import time
import pickle
import h5py


class TinyImageNet(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx, ...]
        y = self.y[idx, ...]

        if self.transform is not None:
            x = self.transform(x)
        return x, y


def load_imagenet(num_train_classes):
  def load_class_images(class_string, label):
    """
    Loads all images in folder class_string.

    :param class_string: image folder (e.g. 'n01774750')
    :param label: label to be assigned to these images
    :return class_k_img: (num_files, width, height, 3) numpy array containing
                         images of folder class_string
    :return class_k_labels: numpy array containing labels
    """
    class_k_path = os.path.join('A:\Datasets\imagenet\\tiny-imagenet-200\\tiny-imagenet-200\\train',
                                class_string, 'images')
    file_list = sorted(os.listdir(class_k_path))

    dtype = np.uint8

    class_k_img = np.zeros((len(file_list), 3, 64, 64), dtype=dtype)
    for l, f in enumerate(file_list):
      file_path = os.path.join('A:\Datasets\imagenet\\tiny-imagenet-200\\tiny-imagenet-200\\train', class_string,
                               'images', f)
      img = mpimg.imread(file_path)
      if len(img.shape) == 2:
        class_k_img[l, :, :, :] = np.expand_dims(img, 0).astype(dtype)
      else:
        img = np.transpose(img, (2, 0, 1))
        class_k_img[l, :, :, :] = img.astype(dtype)

    class_k_labels = label * np.ones(len(file_list), dtype=dtype)

    return class_k_img, class_k_labels

  # get the word description for all imagenet 82115 classes
  all_class_dict = {}
  for k, line in enumerate(open('A:\Datasets\imagenet\\tiny-imagenet-200\\tiny-imagenet-200\\words.txt', 'r')):
    n_id, description = line.split('\t')[:2]
    all_class_dict[n_id] = description

  # this will be the description for our 200 classes
  class_dict = {}

  # we enumerate the classes according to their folder names:
  # 'n01443537' -> 0
  # 'n01629819' -> 1
  # ...
  ls_train = sorted(os.listdir('A:\Datasets\imagenet\\tiny-imagenet-200\\tiny-imagenet-200\\train'))
  img = None
  labels = None
  ood_x = None
  ood_y = None

  # the first num_train_classes will make the training, validation, test sets
  for k in range(num_train_classes):
    # the word descritpion of the current class
    class_dict[k] = all_class_dict[ls_train[k]]
    # load images and labels for current class
    class_k_img, class_k_labels = load_class_images(ls_train[k], k)
    # concatenate all samples and labels
    if img is None:
      img = class_k_img
      labels = class_k_labels
    else:
      img = np.concatenate((img, class_k_img), axis=0)
      labels = np.concatenate((labels, class_k_labels))

  # the remaining classes are the out of domain (ood) set
  for k in range(num_train_classes, 200):
    class_dict[k] = all_class_dict[ls_train[k]]
    class_k_img, class_k_labels = load_class_images(ls_train[k], k)
    if ood_x is None:
      ood_x = class_k_img
      ood_y = class_k_labels
    else:
      ood_x = np.concatenate((ood_x, class_k_img), axis=0)
      ood_y = np.concatenate((ood_y, class_k_labels))

  return img, labels, ood_x, ood_y, class_dict

def split_data(x, y, N):
    x_N = x[0:N, ...]
    y_N = y[0:N]
    x_rest = x[N:, ...]
    y_rest = y[N:, ...]
    return x_N, y_N, x_rest, y_rest


def execute():
    num_train_classes = 190
    print('Loading data...')
    start_time = time.time()
    train_x, train_y, ood_x, ood_y, class_dict = load_imagenet(num_train_classes)
    print('Data loaded in {} seconds.'.format(time.time() - start_time))


    # fix random seed
    np.random.seed(42)

    # shuffle
    N = train_x.shape[0]
    rp = np.random.permutation(N)
    train_x = train_x[rp, ...]
    train_y = train_y[rp]

    # train/validation split 80 - 10 - 10
    N_val = int(round(N * 0.1))
    N_test = int(round(N * 0.1))
    val_x, val_y, train_x, train_y = split_data(train_x, train_y, N_val)
    test_x, test_y, train_x, train_y = split_data(train_x, train_y, N_test)

    # shuffle ood data
    N_ood = ood_x.shape[0]
    rp = np.random.permutation(N_ood)
    ood_x = ood_x[rp, ...]
    ood_y = ood_y[rp]

    # convert all data into float32
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    val_x = val_x.astype(np.float32)
    val_y = val_y.astype(np.float32)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.float32)
    ood_x = ood_x.astype(np.float32)
    ood_y = ood_y.astype(np.float32)

    # normalize
    train_x /= 255.
    val_x /= 255.
    test_x /= 255.
    ood_x /= 255.

    print(train_x.shape)
    print(val_x.shape)
    print(test_x.shape)
    print(ood_x.shape)

    h5f = h5py.File('tinyImgNet.h5', 'w')
    h5f.create_dataset('train_x', data=train_x)
    h5f.create_dataset('train_y', data=train_y)
    h5f.create_dataset('val_x', data=val_x)
    h5f.create_dataset('val_y', data=val_y)
    h5f.create_dataset('test_x', data=test_x)
    h5f.create_dataset('test_y', data=test_y)
    h5f.create_dataset('ood_x', data=ood_x)
    h5f.create_dataset('ood_y', data=ood_y)
    h5f.close()

    with open('class_dict.pickle', 'wb+') as f:
        pickle.dump(class_dict, f)

    return

if __name__ == "__main__":
   execute()