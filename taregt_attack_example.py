#!/usr/bin/env python
# coding: utf-8

# # White-box Attack on CIFAR10

# In[1]:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import sys
sys.path.append("..")
import torch
from torchatt.attacks.ssmiv0315_module2_del import VNIFGSM

from torchvision import models
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from models.resnet import Resnet18

def Mkdir(path):  # path是指定文件夹路径
    if os.path.isdir(path):
        # print('文件夹已存在')
        pass
    else:
        os.makedirs(path)


def numpy_to_variable(image, device=torch.device('cuda:0')):
    x_image = np.expand_dims(image, axis=0)
    x_image = Variable(torch.tensor(x_image), requires_grad=True)
    x_image = x_image.to(device)
    x_image.retain_grad()
    return x_image


def load_image(
        shape=(224, 224), bounds=(0, 1), dtype=np.float32,
        data_format='channels_last', fname='example.png', abs_path=False, fpath=None):
    """ Returns a resized benign_image of target fname.

    Parameters
    ----------
    shape : list of integers
        The shape of the returned benign_image.
    data_format : str
        "channels_first" or "channls_last".

    Returns
    -------
    benign_image : array_like
        The example benign_image in bounds (0, 255) or (0, 1)
        depending on bounds parameter
    """
    if abs_path == True:
        assert fpath is not None, "fpath has not to be None when abs_path is True."
    assert len(shape) == 2
    assert data_format in ['channels_first', 'channels_last']
    if not abs_path:
        path = os.path.join(os.path.dirname(__file__), 'images/%s' % fname)
    else:
        path = fpath
    image = Image.open(path)
    image = image.resize(shape)
    image = image.convert('RGB')
    image = np.asarray(image, dtype=dtype)
    image = image[:, :, :3]
    assert image.shape == shape + (3,)
    if data_format == 'channels_first':
        image = np.transpose(image, (2, 0, 1))
    if bounds != (0, 255):
        image /= 255.
    return image

def cam_one(cam_path_incepetion, cam_path_wide_resnet101, cam_path_resnet34, temp_image_name):

    cam_image_path_incepetion = os.path.join(cam_path_incepetion, temp_image_name)
    cam_image_np_incepetion = load_image(
        data_format='channels_first',
        abs_path=True, fpath=cam_image_path_incepetion)
    cam_image_incepetion = numpy_to_variable(cam_image_np_incepetion)

    cam_image_path_wide_resnet = os.path.join(cam_path_wide_resnet101, temp_image_name)
    cam_image_np_wide_resnet = load_image(
        data_format='channels_first',
        abs_path=True, fpath=cam_image_path_wide_resnet)
    cam_image_wide_resnet101 = numpy_to_variable(cam_image_np_wide_resnet)

    cam_image_path_resnet34 = os.path.join(cam_path_resnet34, temp_image_name)
    cam_image_np_resnet34 = load_image(
        data_format='channels_first',
        abs_path=True, fpath=cam_image_path_resnet34)
    cam_image_resnet34 = numpy_to_variable(cam_image_np_resnet34)

    cam_image_max = torch.max(torch.max(cam_image_wide_resnet101, cam_image_incepetion),
                                        cam_image_resnet34)


    cam_image_min = torch.min(torch.min(cam_image_wide_resnet101, cam_image_incepetion),
                                        cam_image_resnet34)

    cam_image = cam_image_wide_resnet101 + cam_image_incepetion + cam_image_resnet34

    return cam_image_max, cam_image_min, cam_image


if __name__ == '__main__':

    st = [9]
    kc = [4]
    device = "cuda"
    model = models.resnet50(pretrained=True).to(device).eval()
    feature_model = Resnet18()

    tar_image_path = 'target_image/ILSVRC2012_val_00000824.JPEG'
    tar_image_np = load_image(
        data_format='channels_first',
        abs_path=True, fpath=tar_image_path)
    tar_inputs = numpy_to_variable(tar_image_np)
    temp_image_name = 'target_image/ILSVRC2012_val_00000824.JPEG'

    for k in kc:
        dataset_dir = 'benign_image'
        path = 'benign_image'
        cam_path_incepetion = 'CAM/incepetion'
        cam_path_wide_resnet101 = 'CAM/wide_resnet101'
        cam_path_resnet34 = 'CAM/resnet34'

        images_name = os.listdir(path)
        internal = [i for i in range(29)]
        for s in st:
            outpath = 'result/'+str(s)+'/Tar_alpha=1_500_wide_resnet101_incepetion_resnet34_0.75_4'
            Mkdir(outpath)
            # print(outpath)

            atk = VNIFGSM(feature_model, model, eps=16. / 255, alpha=1. / 500, steps=s)
            for batch, temp_image_name in enumerate(tqdm(images_name)):
                #
                # if os.path.isfile(outpath + '/' + temp_image_name):
                #     print("已存在")
                #     continue
                # print("迭代轮次：" + str(s) + ", 攻击第" + str(batch + 1) + "张图像")
                temp_image_path = os.path.join(dataset_dir, temp_image_name)
                image_np = load_image(
                    data_format='channels_first',
                    abs_path=True, fpath=temp_image_path)
                inputs = numpy_to_variable(image_np)

                target_labels = np.argmax(model(tar_inputs).detach().cpu()).unsqueeze(0)

                # cam_image_max, cam_image_min, cam_image = cam(model, dataset_dir, path, cam_path_incepetion,
                #                                               cam_path_wide_resnet101, cam_path_resnet34,
                #                                               target_labels)

                cam_image_max, cam_image_min, cam_image = cam_one(cam_path_incepetion, cam_path_wide_resnet101,
                                                                  cam_path_resnet34, temp_image_name)

                atk.set_mode_targeted_by_label(quiet=True)

                adv_images = atk(inputs, target_labels, tar_inputs, cam_image_max, cam_image_min, 4,
                                 attack_layer_idx=4, internal=internal)

                adv_img = (np.transpose(adv_images[0].cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)

                adv_img = Image.fromarray(adv_img)
                adv_img.save(os.path.join(outpath, temp_image_name))















