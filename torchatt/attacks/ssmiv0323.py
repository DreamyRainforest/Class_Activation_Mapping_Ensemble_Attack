import torch
import torch.nn as nn

from ..ssmi_v2_attack import Attack


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

    def __init__(self, model1, model2, model3, model4, model5, model6, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2):
        super().__init__("VNIFGSM", model1, model2, model3, model4, model5, model6)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels, cam_image):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        cam_image = cam_image.clone().detach().to(self.device)
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
            outputs1, outputs2, outputs3, outputs4, outputs5, outputs6 = self.get_logits(nes_images)

            # Calculate loss
            if self.targeted:
                cost1 = -loss(outputs1, target_labels)
                cost2 = -loss(outputs2, target_labels)
                cost3 = -loss(outputs3, target_labels)
                cost4 = -loss(outputs4, target_labels)
                cost5 = -loss(outputs5, target_labels)
                cost6 = -loss(outputs6, target_labels)
            else:
                cost1 = loss(outputs1, labels)
                cost2 = loss(outputs2, labels)
                cost3 = loss(outputs3, labels)
                cost4 = loss(outputs4, labels)
                cost5 = loss(outputs5, labels)
                cost6 = loss(outputs6, labels)

            # Update adversarial images
            adv_grad1 = torch.autograd.grad(cost1, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            grad1 = (adv_grad1 + v) / torch.mean(torch.abs(adv_grad1 + v), dim=(1, 2, 3), keepdim=True)
            grad1 = grad1 + momentum * self.decay

            adv_grad2 = torch.autograd.grad(cost2, adv_images,
                                            retain_graph=False, create_graph=False)[0]
            grad2 = (adv_grad2 + v) / torch.mean(torch.abs(adv_grad2 + v), dim=(1, 2, 3), keepdim=True)
            grad2 = grad2 + momentum * self.decay


            adv_grad3 = torch.autograd.grad(cost3, adv_images,
                                            retain_graph=False, create_graph=False)[0]
            grad3 = (adv_grad3 + v) / torch.mean(torch.abs(adv_grad3 + v), dim=(1, 2, 3), keepdim=True)
            grad3 = grad3 + momentum * self.decay

            adv_grad4 = torch.autograd.grad(cost4, adv_images,
                                            retain_graph=False, create_graph=False)[0]
            grad4 = (adv_grad4 + v) / torch.mean(torch.abs(adv_grad4 + v), dim=(1, 2, 3), keepdim=True)
            grad4 = grad4 + momentum * self.decay

            adv_grad5 = torch.autograd.grad(cost5, adv_images,
                                            retain_graph=False, create_graph=False)[0]
            grad5 = (adv_grad5 + v) / torch.mean(torch.abs(adv_grad5 + v), dim=(1, 2, 3), keepdim=True)
            grad5 = grad5 + momentum * self.decay

            adv_grad6 = torch.autograd.grad(cost6, adv_images,
                                            retain_graph=False, create_graph=False)[0]
            grad6 = (adv_grad6 + v) / torch.mean(torch.abs(adv_grad6 + v), dim=(1, 2, 3), keepdim=True)
            grad6 = grad6 + momentum * self.decay

            # momentum = torch.max(torch.max(torch.max(grad1, grad2),
            #                               torch.max(grad3, grad4)),
            #                     torch.max(grad5, grad6))

            momentum = (grad1 + grad2 + grad3 + grad4 + grad5 + grad6) / 6

            # adv_grad = torch.max(torch.max(torch.max(adv_grad1, adv_grad2),
            #                                torch.max(adv_grad3, adv_grad4)),
            #                      torch.max(adv_grad5, adv_grad6))
            adv_grad = (adv_grad1 + adv_grad2 + adv_grad3 + adv_grad4 + adv_grad5 + adv_grad6) / 6


            # Calculate Gradient Variance
            GV_grad1 = torch.zeros_like(images).detach().to(self.device)
            GV_grad2 = torch.zeros_like(images).detach().to(self.device)
            GV_grad3 = torch.zeros_like(images).detach().to(self.device)
            GV_grad4 = torch.zeros_like(images).detach().to(self.device)
            GV_grad5 = torch.zeros_like(images).detach().to(self.device)
            GV_grad6 = torch.zeros_like(images).detach().to(self.device)


            for _ in range(self.N):
                neighbor_images = adv_images.detach() + \
                                  torch.randn_like(images).uniform_(-self.eps*self.beta, self.eps*self.beta)
                neighbor_images.requires_grad = True
                outputs1, outputs2, outputs3, outputs4, outputs5, outputs6 = self.get_logits(neighbor_images)

                # Calculate loss
                if self.targeted:
                    cost1 = -loss(outputs1, target_labels)
                    cost2 = -loss(outputs2, target_labels)
                    cost3 = -loss(outputs3, target_labels)
                    cost4 = -loss(outputs4, target_labels)
                    cost5 = -loss(outputs5, target_labels)
                    cost6 = -loss(outputs6, target_labels)

                else:
                    cost1 = loss(outputs1, labels)
                    cost2 = loss(outputs2, labels)
                    cost3 = loss(outputs3, labels)
                    cost4 = loss(outputs4, labels)
                    cost5 = loss(outputs5, labels)
                    cost6 = loss(outputs6, labels)

                GV_grad1 += torch.autograd.grad(cost1, neighbor_images,
                                               retain_graph=False, create_graph=False)[0]
                GV_grad2 += torch.autograd.grad(cost2, neighbor_images,
                                               retain_graph=False, create_graph=False)[0]
                GV_grad3 += torch.autograd.grad(cost3, neighbor_images,
                                               retain_graph=False, create_graph=False)[0]
                GV_grad4 += torch.autograd.grad(cost4, neighbor_images,
                                               retain_graph=False, create_graph=False)[0]
                GV_grad5 += torch.autograd.grad(cost5, neighbor_images,
                                               retain_graph=False, create_graph=False)[0]
                GV_grad6 += torch.autograd.grad(cost6, neighbor_images,
                                               retain_graph=False, create_graph=False)[0]

            # GV_grad  = torch.max(torch.max(torch.max(GV_grad1, GV_grad2),
            #                          torch.max(GV_grad3, GV_grad4)),
            #                torch.max(GV_grad5, GV_grad6))
            #

            GV_grad = (GV_grad1 + GV_grad2 + GV_grad3 + GV_grad4 + GV_grad5 + GV_grad6) / 6
            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad

            # sg = torch.max(torch.max(torch.max(grad1, grad2),
            #                                 torch.max(grad3, grad4)),
            #                       torch.max(grad5, grad6)).sign()

            sg = (grad1 + grad2 + grad3 + grad4 + grad5 + grad6).sign()
            # sg = torch.min(torch.min(torch.min(grad1, grad2), torch.min(grad3, grad4)), torch.min(grad5, grad6)).sign()

            # 相乘效果较差
            # sg = (grad1 * grad2 * grad3 * grad4 * grad5 * grad6).sign()

            adv_images = adv_images.detach() + self.alpha * sg * cam_image
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()



        return adv_images
