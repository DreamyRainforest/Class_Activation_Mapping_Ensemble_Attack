import torch
import torch.nn as nn
from ..attack import Attack
import copy
import torch




class DispersionAttack_gpu_reduce(object):
    """ Dispersion Reduction (DR) attack, using pytorch."""
    def __init__(self, model, epsilon=16. / 255, step_size=2/255, steps=10):

        self.step_size = step_size
        self.epsilon = epsilon
        self.steps = steps
        self.model = copy.deepcopy(model)


    def __call__(self, X_nat_var, seg_image, temp_image_name, attack_layer_idx=-1, internal=[]):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()


        # 原图
        X_var= torch.clone(X_nat_var)
        seg_var = torch.clone(seg_image)

        for i in range(self.steps):
            X_var.retain_grad()
            internal_features, pred = self.model.prediction(X_var, internal=internal)
            logit = internal_features[attack_layer_idx]

            loss = -1 * logit.std()
            self.model.zero_grad()
            loss.backward()
            grad = X_var.grad.data
            # X_var = X_var.detach() - self.step_size * grad.sign_()
            X_var = X_var.detach() - self.step_size * grad.sign_() * seg_var
            X_var = torch.max(torch.min(X_var, X_nat_var + self.epsilon), X_nat_var - self.epsilon)
            X_var = torch.clamp(X_var, 0, 1)

        return X_var.detach()






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

    def forward(self, images, labels, seg_image, temp_image_name, attack_layer_idx=-1, internal=[]):
        r"""
        Overridden.
        """
        # dr 攻击
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()

        # 原图
        X_var = torch.clone(images)
        seg_var = torch.clone(seg_image)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.steps):

            X_var.retain_grad()
            internal_features, pred = self.net.prediction(X_var, internal=internal)
            logit = internal_features[attack_layer_idx]

            loss_DR = -1 * logit.std()
            self.net.zero_grad()
            loss_DR.backward()
            grad_DR = X_var.grad.data

            # 带增量的DR
            grad_DR = grad_DR / torch.mean(torch.abs(grad_DR), dim=(1, 2, 3), keepdim=True)
            grad_DR = grad_DR + momentum
            momentum = grad_DR

            X_var = X_var.detach() - self.alpha * grad_DR.sign_()*seg_var
            X_var = torch.max(torch.min(X_var, images + self.eps), images - self.eps)
            X_var = torch.clamp(X_var, 0, 1)

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

            # adv_images = adv_images.detach() + self.alpha*grad.sign()*seg_var
            adv_images = adv_images.detach() + self.gama * self.alpha * grad.sign() * seg_var - self.gamaa * self.alpha * grad_DR.sign_() * seg_var
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
