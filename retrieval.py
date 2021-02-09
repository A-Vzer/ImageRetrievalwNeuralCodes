from data import execute, TinyImageNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import NeuralCodes
import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt

with h5py.File('tinyImgNet.h5', "r") as f:
    a_group_key = list(f.keys())[0]
    # Get the data
    ood_x = torch.from_numpy(f['ood_x'][()])
    ood_y = torch.from_numpy(f['ood_y'][()])

with open('class_dict.pickle', 'rb') as f:
    class_dict = pickle.load(f)


def get_codes():
    bs = 256
    model = NeuralCodes().cuda()
    model.load_state_dict(torch.load('saves/model.pt')['model'])
    torch.cuda.empty_cache()
    ood_data = TinyImageNet(x=ood_x, y=ood_y)
    ood_loader = DataLoader(ood_data, batch_size=bs, shuffle=True)
    for idx, (image, target) in enumerate(ood_loader):
        model.eval()
        image = image.cuda()
        x1, x2, x3 = model.get_codes(image)
        if idx == 0:
            code1 = x1.detach().cpu().numpy()
            code2 = x2.detach().cpu().numpy()
            code3 = x3.detach().cpu().numpy()
        else:
            code1 = np.concatenate((code1, x1.detach().cpu().numpy()), 0)
            code2 = np.concatenate((code2, x2.detach().cpu().numpy()), 0)
            code3 = np.concatenate((code3, x3.detach().cpu().numpy()), 0)

    code1 = code1 / np.linalg.norm(code1, axis=1, keepdims=True)
    code2 = code2 / np.linalg.norm(code2, axis=1, keepdims=True)
    code3 = code3 / np.linalg.norm(code3, axis=1, keepdims=True)

    distances1 = np.ones((code1.shape[0], code1.shape[0])) - np.matmul(code1, np.transpose(code1))
    distances2 = np.ones((code2.shape[0], code2.shape[0])) - np.matmul(code2, np.transpose(code2))
    distances3 = np.ones((code3.shape[0], code3.shape[0])) - np.matmul(code3, np.transpose(code3))

    return [distances1, distances2, distances3]


def average_precision(sorted_class_vals, true_class):
    ind = [True if sorted_class_vals[i] == true_class else False for i in range(len(sorted_class_vals))]
    num_positive = np.sum(ind)
    if num_positive == 0:
        return 0
    cum_ind = np.cumsum(ind).astype(np.float32)
    enum = np.array(range(1, len(ind)+1)).astype(np.float32)
    return np.sum(cum_ind * ind / enum) / num_positive


def frame_image(img, frame_width):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution/number of pixels in x and y
    if img.ndim == 3: # rgb or rgba array
        framed_img = np.ones((b+ny+b, b+nx+b, img.shape[2]))
    elif img.ndim == 2: # grayscale image
        framed_img = np.ones((b+ny+b, b+nx+b))
    framed_img[b:-b, b:-b] = img
    return framed_img


def show_images(index, data, images, labels):
    f, axarr = plt.subplots(1, 6, figsize=(15, 15))
    true_class = class_dict[labels[index]]
    print("{}: {}".format(index, true_class))
    axarr[0].imshow(np.transpose(images[index, :, :, :], (1, 2, 0)), interpolation='nearest')  # real image
    for k in range(data.shape[0] - 1):
        idx = data[k + 1]
        if class_dict[labels[idx]] == true_class:  # if same class show white border
            axarr[k + 1].imshow(frame_image(np.transpose(images[idx, :, :, :], (1, 2, 0)), 3), interpolation='nearest')
        else:
            axarr[k + 1].imshow(np.transpose(images[idx, :, :, :], (1, 2, 0)), interpolation='nearest')

    plt.show()


def retrieve(distances, l=1):
    aps = 0
    x = ood_x.numpy()
    y = ood_y.numpy()
    for i in range(x.shape[0]):
        images = distances[l][i, :]  # get distances for query to gallery images
        images = images.argsort()  # get indexes for sorting
        truth = class_dict[y[images[0]]]  # get first element of sorted indices
        series = [class_dict[y[i]] for i in images[1:]]
        ap = average_precision(series, truth)
        aps += ap
        if i < 10:
            show_images(i, images[:6], x, y)
            print(f"AP: {ap}")
            print(f"mAP layer {l}: {aps/y.shape[0]}")


if __name__ == "__main__":
    distances = get_codes()
    retrieve(distances, l=0)
