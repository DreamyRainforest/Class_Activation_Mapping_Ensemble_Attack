
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.models as models
import csv
from tqdm import tqdm

import numpy as np

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



if __name__ == '__main__':


    dataset_dir = 'benign_image'
    path = 'benign_image'
    images_name = os.listdir(path)
    # target
    # adv_dir = 'result/9/Tar_alpha=1_500_wide_resnet101_incepetion_resnet34_0.75_4'
    # untaregt
    adv_dir = 'result/10/alpha=500_wide_resnet101_incepetion_resnet34_0.75_2'
    tar_image_path = 'target_image/ILSVRC2012_val_00000824.JPEG'
    tar_image_np = load_image(
        data_format='channels_first',
        abs_path=True, fpath=tar_image_path)
    tar_inputs = numpy_to_variable(tar_image_np)

    model = models.resnet50(pretrained=True)
    model = model.eval()
    model.cuda()

    for idx, temp_image_name in enumerate(tqdm(images_name)):
        temp_image_path = os.path.join(dataset_dir, temp_image_name[:-4] + ".jpg")
        adv_image_path = os.path.join(adv_dir, temp_image_name[:-4] + ".jpg")

        image_np = load_image(
            data_format='channels_first',
            abs_path=True, fpath=temp_image_path)
        image = numpy_to_variable(image_np)

        adv_image_np = load_image(
            data_format='channels_first',
            abs_path=True, fpath=adv_image_path)
        adv_image = numpy_to_variable(adv_image_np)

        or_label = np.argmax(model(image).detach().cpu().numpy())
        adv_label = np.argmax(model(adv_image).detach().cpu().numpy())

        target_labels = np.argmax(model(tar_inputs).detach().cpu()).unsqueeze(0)
        # untarget
        print(f'name: {temp_image_name}, or_label: {or_label}, adv_label: {adv_label}')
        # target
        # print(f'name: {temp_image_name}, or_label: {or_label}, adv_label: {adv_label}, target_labels: {target_labels}')





