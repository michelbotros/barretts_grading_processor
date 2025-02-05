import os
import torch
import segmentation_models_pytorch as smp
import yaml
from preprocessing import get_preprocessing
import numpy as np


def load_config(user_config):
    with open(user_config, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    return data['wholeslidedata'], data['net']


def load_trained_segmentation_model(exp_dir, model_path):
    """ Loads the trained model.

    Args:
        exp_dir: directory that hold all the information from an experiments (src, checkpoints)
        model_path: path to the trained model

    """
    user_config = os.path.join(exp_dir, 'src/configs/base_config.yml')
    _, train_config = load_config(user_config)

    # LOAD MODEL
    model = load_segmentation_model(train_config, activation=None)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('Loaded model from {}'.format(model_path))

    # LOAD PREPROCESSING
    if train_config['encoder_weights']:
        preprocessing = get_preprocessing(smp.encoders.get_preprocessing_fn(
            train_config['encoder_name'], train_config['encoder_weights']))
    else:
        preprocessing = get_preprocessing()
    print('During training we used {} as encoder with weights from {}.'.format(train_config['encoder_name'],
                                                                               train_config['encoder_weights']))
    return model, preprocessing


def load_segmentation_model(train_config, activation=None):
    """ Loads the segmentation model.

    Args:
        train_config: config that specifies which model to train.
        activation: activation applied at the end

    """
    # print('Loading model: {}, weights: {}'.format(train_config['segmentation_model'], train_config['encoder_weights']))

    if train_config['segmentation_model'] == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=train_config['encoder_name'],
            encoder_weights=None,
            in_channels=train_config['n_channels'],
            classes=train_config['n_classes'],
            activation=activation
        )
    elif train_config['segmentation_model'] == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=train_config['encoder_name'],
            encoder_weights=None,
            in_channels=train_config['n_channels'],
            classes=train_config['n_classes'],
            activation=activation
        )
    else:
        model = smp.Unet(
            encoder_name=train_config['encoder_name'],
            encoder_weights=None,
            in_channels=train_config['n_channels'],
            classes=train_config['n_classes'],
            activation=activation
        )

    return model


def load_model(model_path, train_config, device=None):
    """ Load a model

    """
    model = load_segmentation_model(train_config, activation=None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print('Loading model from: {}'.format(model_path))

    return model


class SingleModel:
    def __init__(self, model):
        self.model = model

    def forward(self, x, y):
        """
        Args:
            x:
                torch.tensor: (B, H, W)
            y:

        Returns:
            y_avg_prob:
                np.array (B, C, H, W)
        """

        y_pred_batch = self.model(x)
        y_avg_prob = torch.nn.functional.softmax(y_pred_batch, dim=1).cpu().detach().numpy()

        return y_avg_prob


class Ensemble:
    """ An ensemble of models.

    """

    def __init__(self, ensemble_exp_dir, device, m=5):
        # load base config
        self.user_config = os.path.join(ensemble_exp_dir, 'base_config.yml')
        _, self.train_config = load_config(self.user_config)
        self.nets = sorted(
            [os.path.join(ensemble_exp_dir, n, 'checkpoints', 'best_model.pt') for n in os.listdir(ensemble_exp_dir) if
             'net' in n])[:m]
        self.device = device
        self.models = [load_model(n, self.train_config, self.device) for n in self.nets]

    def forward(self, x, y):
        """ Forwards a batch of images through the ensemble.

        Args:
            x:
                torch.tensor: (B, H, W)
            y:

        Returns:
            y_avg_prob:
                np.array (B, C, H, W)
        """

        # for averaging the M predictions
        y_pred_patches = []  # elements are (M, C, H, W)

        for i, model in enumerate(self.models):
            # forward, convert logits to probabilities
            y_pred_batch_m = model(x.to(self.device))
            y_pred_batch_m_soft = torch.nn.functional.softmax(y_pred_batch_m, dim=1).cpu().detach().numpy()
            y_pred_patches.append(y_pred_batch_m_soft[:len(y)])

        # stack patches: (M, B, C, H, W)
        y_pred_stack = np.stack(y_pred_patches, axis=0)

        # average probabilities: (B, C, H, W)
        y_avg_prob = np.mean(y_pred_stack, axis=0)

        return y_avg_prob
