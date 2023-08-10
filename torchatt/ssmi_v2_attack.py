import time
import logging
from collections import OrderedDict
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset


def wrapper_method(func):
    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get('_attacks').values():
            eval("atk." + func.__name__ + "(*args, **kwargs)")
        return result

    return wrapper_func


class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_model_training_mode`.
    """

    def __init__(self, name, model1, model2, model3, model4, model5, model6):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self._attacks = OrderedDict()

        self.set_model(model1, model2, model3, model4, model5, model6)
        self.device = next(model1.parameters()).device

        # Controls attack mode.
        self.attack_mode = 'default'
        self.supported_mode = ['default']
        self.targeted = False
        self._target_map_function = None

        # Controls when normalization is used.
        self.normalization_used = None
        self._normalization_applied = None
        self._set_auto_normalization_used(model1, model2, model3, model4, model5, model6)

        # Controls model mode during attack.
        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, inputs, labels=None, *args, **kwargs):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    @wrapper_method
    def set_model(self, model1, model2, model3, model4, model5, model6):
        self.model1 = model1
        self.model_name1 = str(model1).split("(")[0]

        self.model2 = model2
        self.model_name2 = str(model2).split("(")[0]

        self.model3 = model3
        self.model_name3 = str(model3).split("(")[0]

        self.model4 = model4
        self.model_name4= str(model4).split("(")[0]

        self.model5 = model5
        self.model_name5 = str(model5).split("(")[0]

        self.model6 = model6
        self.model_name6 = str(model6).split("(")[0]

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits1 = self.model1(inputs)

        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits2 = self.model2(inputs)

        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits3 = self.model3(inputs)

        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits4 = self.model4(inputs)

        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits5 = self.model5(inputs)

        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits6 = self.model6(inputs)

        return logits1, logits2, logits3, logits4, logits5, logits6

    @wrapper_method
    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag

    @wrapper_method
    def set_device(self, device):
        self.device = device

    @wrapper_method
    def _set_auto_normalization_used(self, model1, model2, model3, model4, model5, model6):
        mean = getattr(model1, 'mean', None)
        std = getattr(model1, 'std', None)
        if (mean is not None) and (std is not None):
            if isinstance(mean, torch.Tensor):
                mean = mean.cpu().numpy()
            if isinstance(std, torch.Tensor):
                std = std.cpu().numpy()
            if (mean != 0).all() or (std != 1).all():
                self.set_normalization_used(mean, std)

    #                 logging.info("Normalization automatically loaded from `model.mean` and `model.std`.")

    @wrapper_method
    def set_normalization_used(self, mean, std):
        self.normalization_used = {}
        n_channels = len(mean)
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used['mean'] = mean
        self.normalization_used['std'] = std
        self._normalization_applied = True

    def normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return inputs * std + mean

    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self.attack_mode

    @wrapper_method
    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        self.attack_mode = 'default'
        self.targeted = False
        print("Attack mode is changed to 'default.'")

    @wrapper_method
    def _set_mode_targeted(self, mode):
        if "targeted" not in self.supported_mode:
            raise ValueError("Targeted mode is not supported.")
        self.targeted = True
        self.attack_mode = mode
        print("Attack mode is changed to '%s'." % mode)

    @wrapper_method
    def set_mode_targeted_by_function(self, target_map_function):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda inputs, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)

        """
        self._set_mode_targeted('targeted(custom)')
        self._target_map_function = target_map_function

    @wrapper_method
    def set_mode_targeted_random(self):
        r"""
        Set attack mode as targeted with random labels.
        Arguments:
            num_classses (str): number of classes.

        """
        self._set_mode_targeted('targeted(random)')
        self._target_map_function = self.get_random_target_label

    @wrapper_method
    def set_mode_targeted_least_likely(self, kth_min=1):
        r"""
        Set attack mode as targeted with least likely labels.
        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)

        """
        self._set_mode_targeted('targeted(least-likely)')
        assert (kth_min > 0)
        self._kth_min = kth_min
        self._target_map_function = self.get_least_likely_label

    @wrapper_method
    def set_model_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    @wrapper_method
    def _change_model_mode(self, given_training):
        if self._model_training:

            self.model1.train()
            for _, m in self.model1.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()

            self.model2.train()
            for _, m in self.model2.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()

            self.model3.train()
            for _, m in self.model3.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()

            self.model4.train()
            for _, m in self.model4.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()

            self.model5.train()
            for _, m in self.model5.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()

            self.model6.train()
            for _, m in self.model6.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()


        else:
            self.model1.eval()
            self.model2.eval()
            self.model3.eval()
            self.model4.eval()
            self.model5.eval()
            self.model6.eval()

    @wrapper_method
    def _recover_model_mode(self, given_training):
        if given_training:
            self.model1.train()
            self.model2.train()
            self.model3.train()
            self.model4.train()
            self.model5.train()
            self.model6.train()

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False,
             save_predictions=False, save_clean_inputs=False, save_type='float'):
        r"""
        Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_predictions (bool): True for saving predicted labels (Default: False)
            save_clean_inputs (bool): True for saving clean inputs (Default: False)

        """
        if save_path is not None:
            adv_input_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_inputs:
                input_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)
        given_training1 = self.model1.training
        given_training2 = self.model2.training
        given_training3 = self.model3.training
        given_training4 = self.model4.training
        given_training5 = self.model5.training
        given_training6 = self.model6.training


        for step, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            adv_inputs = self.__call__(inputs, labels)
            batch_size = len(inputs)

            if verbose or return_verbose:
                with torch.no_grad():
                    outputs = self.get_output_with_eval_nograd(adv_inputs)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (pred == labels.to(self.device))
                    correct += right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate l2 Perdistance
                    delta = (adv_inputs - inputs.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))
                    l2 = torch.cat(l2_distance).mean().item()

                    # Calculate time computation
                    progress = (step + 1) / total_batch * 100
                    end = time.time()
                    elapsed_time = end - start

                    if verbose:
                        self._save_print(progress, rob_acc, l2, elapsed_time, end='\r')

            if save_path is not None:
                adv_input_list.append(adv_inputs.detach().cpu())
                label_list.append(labels.detach().cpu())

                adv_input_list_cat = torch.cat(adv_input_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                save_dict = {'adv_inputs': adv_input_list_cat, 'labels': label_list_cat}

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict['preds'] = pred_list_cat

                if save_clean_inputs:
                    input_list.append(inputs.detach().cpu())
                    input_list_cat = torch.cat(input_list, 0)
                    save_dict['clean_inputs'] = input_list_cat

                if self.normalization_used is not None:
                    save_dict['adv_inputs'] = self.inverse_normalize(save_dict['adv_inputs'])
                    if save_clean_inputs:
                        save_dict['clean_inputs'] = self.inverse_normalize(save_dict['clean_inputs'])

                if save_type == 'int':
                    save_dict['adv_inputs'] = self.to_type(save_dict['adv_inputs'], 'int')
                    if save_clean_inputs:
                        save_dict['clean_inputs'] = self.to_type(save_dict['clean_inputs'], 'int')

                save_dict['save_type'] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end='\n')

        if given_training1:
            self.model1.train()
        if given_training2:
            self.model2.train()

        if given_training3:
            self.model3.train()

        if given_training4:
            self.model4.train()

        if given_training5:
            self.model5.train()

        if given_training6:
            self.model6.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    @staticmethod
    def to_type(inputs, type):
        r"""
        Return inputs as int if float is given.
        """
        if type == 'int':
            if isinstance(inputs, torch.FloatTensor) or isinstance(inputs, torch.cuda.FloatTensor):
                return (inputs * 255).type(torch.uint8)
        elif type == 'float':
            if isinstance(inputs, torch.ByteTensor) or isinstance(inputs, torch.cuda.ByteTensor):
                return inputs.float() / 255
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")
        return inputs

    @staticmethod
    def _save_print(progress, rob_acc, l2, elapsed_time, end):
        print('- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t' \
              % (progress, rob_acc, l2, elapsed_time), end=end)

    @staticmethod
    def load(load_path, batch_size=128, shuffle=False, normalize=None,
             load_predictions=False, load_clean_inputs=False):
        save_dict = torch.load(load_path)
        keys = ['adv_inputs', 'labels']

        if load_predictions:
            keys.append('preds')
        if load_clean_inputs:
            keys.append('clean_inputs')

        if save_dict['save_type'] == 'int':
            save_dict['adv_inputs'] = save_dict['adv_inputs'].float() / 255
            if load_clean_inputs:
                save_dict['clean_inputs'] = save_dict['clean_inputs'].float() / 255

        if normalize is not None:
            n_channels = len(normalize['mean'])
            mean = torch.tensor(normalize['mean']).reshape(1, n_channels, 1, 1)
            std = torch.tensor(normalize['std']).reshape(1, n_channels, 1, 1)
            save_dict['adv_inputs'] = (save_dict['adv_inputs'] - mean) / std
            if load_clean_inputs:
                save_dict['clean_inputs'] = (save_dict['clean_inputs'] - mean) / std

        adv_data = TensorDataset(*[save_dict[key] for key in keys])
        adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=shuffle)
        print("Data is loaded in the following order: [%s]" % (", ".join(keys)))
        return adv_loader

    @torch.no_grad()
    def get_output_with_eval_nograd(self, inputs):
        given_training1 = self.model1.training
        if given_training1:
            self.model1.eval()
        outputs1 = self.get_logits(inputs)
        if given_training1:
            self.model1.train()

        given_training2 = self.model2.training
        if given_training2:
            self.model2.eval()
        outputs2 = self.get_logits(inputs)
        if given_training2:
            self.model2.train()

        given_training3 = self.model3.training
        if given_training3:
            self.model3.eval()
        outputs3 = self.get_logits(inputs)
        if given_training3:
            self.model3.train()

        given_training4 = self.model4.training
        if given_training4:
            self.model4.eval()
        outputs4 = self.get_logits(inputs)
        if given_training4:
            self.model4.train()

        given_training5 = self.model5.training
        if given_training5:
            self.model5.eval()
        outputs5 = self.get_logits(inputs)
        if given_training5:
            self.model5.train()

        given_training6 = self.model6.training
        if given_training6:
            self.model6.eval()
        outputs6 = self.get_logits(inputs)
        if given_training6:
            self.model6.train()

        return outputs1, outputs2, outputs3, outputs4, outputs5, outputs6

    def get_target_label(self, inputs, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function is None:
            raise ValueError('target_map_function is not initialized by set_mode_targeted.')
        target_labels = self._target_map_function(inputs, labels)
        return target_labels

    @torch.no_grad()
    def get_least_likely_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_random_target_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l) * torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    def __call__(self, inputs, labels=None, *args, **kwargs):
        given_training1 = self.model1.training
        self._change_model_mode(given_training1)

        if self._normalization_applied is True:
            inputs1 = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)

            adv_inputs1 = self.forward(inputs1, labels, *args, **kwargs)
            adv_inputs1 = self.normalize(adv_inputs1)
            self._set_normalization_applied(True)
        else:
            adv_inputs1 = self.forward(inputs, labels, *args, **kwargs)

        self._recover_model_mode(given_training1)

        given_training2 = self.model2.training
        self._change_model_mode(given_training2)

        if self._normalization_applied is True:
            inputs2 = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)

            adv_inputs2 = self.forward(inputs2, labels, *args, **kwargs)
            adv_inputs2 = self.normalize(adv_inputs2)
            self._set_normalization_applied(True)
        else:
            adv_inputs2 = self.forward(inputs, labels, *args, **kwargs)

        self._recover_model_mode(given_training2)

        given_training3 = self.model3.training
        self._change_model_mode(given_training1)

        if self._normalization_applied is True:
            inputs3 = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)

            adv_inputs3 = self.forward(inputs3, labels, *args, **kwargs)
            adv_inputs3 = self.normalize(adv_inputs3)
            self._set_normalization_applied(True)
        else:
            adv_inputs3 = self.forward(inputs, labels, *args, **kwargs)

        self._recover_model_mode(given_training3)

        given_training4 = self.model4.training
        self._change_model_mode(given_training4)

        if self._normalization_applied is True:
            inputs4 = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)

            adv_inputs4 = self.forward(inputs4, labels, *args, **kwargs)
            adv_inputs4 = self.normalize(adv_inputs4)
            self._set_normalization_applied(True)
        else:
            adv_inputs4 = self.forward(inputs, labels, *args, **kwargs)

        self._recover_model_mode(given_training4)

        given_training5 = self.model5.training
        self._change_model_mode(given_training5)

        if self._normalization_applied is True:
            inputs5 = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)

            adv_inputs5 = self.forward(inputs5, labels, *args, **kwargs)
            adv_inputs5 = self.normalize(adv_inputs5)
            self._set_normalization_applied(True)
        else:
            adv_inputs5 = self.forward(inputs, labels, *args, **kwargs)

        self._recover_model_mode(given_training5)

        given_training6 = self.model6.training
        self._change_model_mode(given_training6)

        if self._normalization_applied is True:
            inputs6 = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)

            adv_inputs6 = self.forward(inputs6, labels, *args, **kwargs)
            adv_inputs6 = self.normalize(adv_inputs6)
            self._set_normalization_applied(True)
        else:
            adv_inputs6 = self.forward(inputs, labels, *args, **kwargs)

        self._recover_model_mode(given_training6)


        return adv_inputs1, adv_inputs2, adv_inputs3, adv_inputs4, adv_inputs5, adv_inputs6

    def __repr__(self):
        info = self.__dict__.copy()

        del_keys = ['model', 'attack', 'supported_mode']

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info['attack_mode'] = self.attack_mode
        info['normalization_used'] = True if self.normalization_used is not None else False

        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

        attacks = self.__dict__.get('_attacks')

        # Get all items in iterable items.
        def get_all_values(items, stack=[]):
            if (items not in stack):
                stack.append(items)
                if isinstance(items, list) or isinstance(items, dict):
                    if isinstance(items, dict):
                        items = (list(items.keys()) + list(items.values()))
                    for item in items:
                        yield from get_all_values(item, stack)
                else:
                    if isinstance(items, Attack):
                        yield items
            else:
                if isinstance(items, Attack):
                    yield items

        for num, value in enumerate(get_all_values(value)):
            attacks[name + "." + str(num)] = value
            for subname, subvalue in value.__dict__.get('_attacks').items():
                attacks[name + "." + subname] = subvalue
