import torch
import torch.nn as nn
import numpy as np
from ..attack import Attack
import copy
def extract_cam(input_tensor, model):

    with XGradCAM(model) as cam_extractor:
        # Preprocess your data and feed it to the model
        out = model(input_tensor.unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        plt.imshow(activation_map[0].squeeze(0).cpu().numpy())
        plt.axis('off')
        plt.savefig('/data/zhangrui/AdvSmo/adversarial-attacks-pytorch-master/SSMI_modify_v2/result/cam/cam.jpg', dpi=300, bbox_inches='tight',
                    pad_inches=-0.01)


class VNIFGSM(Attack):
    r"""
    VNI-FGSM in the paper 'Enhancing the Transferability of Adversarial Attacks through Variance Tuning
    [https://arxiv.org/abs/2103.15571], Published as a conference paper at CVPR 2021
    Modified from "https://github.com/JHL-HUST/VT"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        N (int): the number of sampled examples in the neighborhood. (Default: 5)
        beta (float): the upper bound of neighborhood. (Default: 3/2)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.VNIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, feature_model, model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2):
        super().__init__("VNIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
        self.supported_mode = ['default', 'targeted']
        self.feature_model = copy.deepcopy(feature_model)

    def forward(self, images, labels, cam_image, cam_image_max, cam_image_min, k, attack_layer_idx=-1, internal=[]):
        r"""
        Overridden.
        """


        for p in self.feature_model.parameters():
            p.requires_grad = False
        self.feature_model.eval()


        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        v = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_images = adv_images + self.decay * self.alpha * momentum

            internal_features, pred = self.feature_model.prediction(images, internal=internal)
            logit = internal_features[attack_layer_idx]

            log = -1 * logit.std()

            outputs = self.get_logits(nes_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels) - torch.norm(cam_image_max-images)
            else:
                cost = loss(outputs, labels) + torch.norm(cam_image_max-images)
                # cost = torch.norm(cam_image_max-images) + torch.norm(cam_image_min - images)
                # cost = loss(outputs, labels) + torch.norm(cam_image - images)
                # cost = 0.5 * loss(outputs, labels) + 1.5 * torch.norm((cam_image_max - images), np.inf)

            # Update adversarial images
            adv_grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v), dim=(1,2,3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + \
                                  torch.randn_like(images).uniform_(-self.eps*self.beta, self.eps*self.beta)
                neighbor_images.requires_grad = True
                outputs = self.get_logits(neighbor_images)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
                GV_grad += torch.autograd.grad(cost, neighbor_images,
                                               retain_graph=False, create_graph=False)[0]
            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad
            # adv_images = adv_images.detach() + self.alpha*grad.sign()
            adv_images = adv_images.detach() + self.alpha * 0.75 * grad.sign() * k * cam_image_max
            # adv_images = adv_images.detach() + self.alpha * 0.75 * grad.sign() * k *cam_image_max + self.alpha * 0.75 * grad.sign() * 2 * (1-cam_image_max)
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
