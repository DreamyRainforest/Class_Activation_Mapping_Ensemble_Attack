import torch
import torch.nn as nn
from ..attack import Attack
import copy
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import os

def numpy_to_variable(image, device=torch.device('cuda:0')):
    x_image = np.expand_dims(image, axis=0)
    x_image = Variable(torch.tensor(x_image), requires_grad=True)
    x_image = x_image.to(device)
    x_image.retain_grad()
    return x_image

class modify_MIFGSM(Attack):

    def __init__(self, model, net, gama=0.5, gamaa=0.5, eps=8/255, alpha=2/255, steps=10, decay=1.0):
        super().__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ['default', 'targeted']
        self.gama = gama
        self.gamaa = gamaa
        self.net = copy.deepcopy(net)
        self.targeted = False



    def forward(self, images, labels, temp_image_name, attack_layer_idx=-1, internal=[]):
        r"""
        Overridden.
        """
        # dr 攻击
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()

        # 原图
        X_var = torch.clone(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        ma = torch.zeros_like(images)
        mask = numpy_to_variable(ma.squeeze(0).detach().cpu().numpy())


        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.steps):

            # MIFGSM
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            si = grad.data
            for i in range(si.shape[0]):
                for j in range(si.shape[1]):
                    if si[:, :, i, j][0][0].detach() > 0 or si[:, :, i, j][0][1].detach()> 0 or si[:, :, i, j][0][2].detach() > 0:
                        mask[:, :, i, j][0][0] = 1
                        mask[:, :, i, j][0][1] = 1
                        mask[:, :, i, j][0][2] = 1

            ama = (np.transpose(mask[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
            mas = Image.fromarray(ama)
            mas.save(os.path.join('/data/zhangrui/AdvSmo/adversarial-attacks-pytorch-master/result/mask', 'mask.jpg'))

            X_var.retain_grad()
            internal_features, pred = self.net.prediction(X_var, internal=internal)
            logit = internal_features[attack_layer_idx]

            loss_DR = -1 * logit.std()
            self.net.zero_grad()
            loss_DR.backward()
            grad_DR = X_var.grad.data
            adv_images = X_var.detach() - self.alpha * grad_DR.sign_() * mask
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
