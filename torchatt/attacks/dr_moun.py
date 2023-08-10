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
        momentum = torch.zeros_like(X_var).detach()
        for i in range(self.steps):
            X_var.retain_grad()
            internal_features, pred = self.model.prediction(X_var, internal=internal)
            logit = internal_features[attack_layer_idx]

            loss = -1 * logit.std()
            self.model.zero_grad()
            loss.backward()
            grad = X_var.grad.data

            grad = grad/ torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad= grad + momentum
            momentum = grad
            # X_var = X_var.detach() - self.step_size * grad.sign_()
            X_var = X_var.detach() - self.step_size * grad.sign_() * seg_var
            X_var = torch.max(torch.min(X_var, X_nat_var + self.epsilon), X_nat_var - self.epsilon)
            X_var = torch.clamp(X_var, 0, 1)


        return X_var.detach()




