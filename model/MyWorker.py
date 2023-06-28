# Worker script that defines the problem to be optimised by BOHB.
# From: https://automl.github.io/HpBandSter/build/html/index.html

import ConfigSpace as CS
from hpbandster.core.worker import Worker

import autoencoders


def convert_hyper_params(config_dict):
    """
    A utility function to convert the hyperparams used for the optimisation.
    It is more efficient to log-sample some of the search dims, which needs to be converted.

    Args:
        config_dict : dictionary containing the sampled configurations by the optimizer

    Returns:
        out_dict : A pre-computed dict in linear scale with the same fields as the input.
        """

    out_dict = {}
    for (key, val) in config_dict.items():
        if key.startswith('log2_'):
            out_dict['_'.join(key.split('_')[1:])] = 2 ** val
        elif key.startswith('log10_'):
            out_dict['_'.join(key.split('_')[1:])] = 10 ** val
        else:
            out_dict[key] = val

    return out_dict


class MyWorker(Worker):
    """
    Worker object for hpbandster to execute
    Defines the task and config parameter space

    Parameters
    ----------
    data_table : str
        Filename containing the entire training dataset
        File should contain an ndarray of shape (`n_features`,`n_samples`)

    nameserver : str
        IP address to the hpbandster nameserver, ex: '127.0.0.1'. The address
        should match the address provided to the constructor of the NameServer
        in your script.

    run_id : str
        Name coined for your current hpbanster run, such as 'best_ever_project'. Arbitrary.
    """

    def __init__(self, Xfile, Yfile, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Xfile = Xfile
        self.Yfile = Yfile

    def compute(self, config, budget, **kwargs):
        """
        A ML-related objective function to optimise via chosen algorithm (e.g. BOHB)

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        conf = convert_hyper_params(config)

        # Option 1: Dimensionality reduction with BOHB-determined hyperparams
        # This solves for argmin(validation loss of AE)
        val_loss = autoencoders.train_modality(Xfile=self.Xfile, Yfile=self.Yfile, max_training_duration=budget, **conf)

        # Option 2: we can use the performance of the classifier as the cost function, instead.
        # This solves for argmin(1-AUC of SVM classifier)
        # There might be OVERFITTING, one needs a validation set.
        # val_loss = autoencoders.train_modality(Xfile=self.Xfile, Yfile=self.Yfile, max_training_duration=budget,
        #                                       classifiers_to_train=['svm'], **conf)

        return ({
                 'loss': float(val_loss),  # this is the a mandatory field to run hyperband
                 'info': val_loss  # can be used for any user-defined information - also mandatory
                })

    @staticmethod
    def get_configspace():
        config = CS.ConfigurationSpace()

        # Common hyperparams
        AE_type = CS.Categorical('AE_type', ['CAE', 'SAE', 'VAE'])
        log10_gradient_threshold = CS.UniformIntegerHyperparameter('log10_gradient_threshold', lower=-6, upper=1)
        seed = CS.UniformIntegerHyperparameter('seed', lower=1, upper=10)

        # CAE-specific hyperparamms
        num_internal_layers = CS.UniformIntegerHyperparameter('num_internal_layers', lower=1, upper=2)
        log2_num_filters = CS.UniformIntegerHyperparameter('log2_num_filters', lower=1, upper=6)
        use_2D_kernels = CS.Categorical('use_2D_kernels', [True, False])
        cond_1 = CS.EqualsCondition(num_internal_layers, AE_type, 'CAE')
        cond_2 = CS.EqualsCondition(log2_num_filters, AE_type, 'CAE')
        cond_3 = CS.EqualsCondition(use_2D_kernels, AE_type, 'CAE')

        # SAE/DAE and VAE only
        log2_latent_dims = CS.UniformIntegerHyperparameter('log2_latent_dims', lower=2, upper=10)
        cond_4 = CS.NotEqualsCondition(log2_latent_dims, AE_type, 'CAE')

        # Add all parameters and constraints to the configuration space
        config.add_hyperparameters([AE_type, log10_gradient_threshold, seed, num_internal_layers,
                                    log2_num_filters, use_2D_kernels, log2_latent_dims])
        config.add_conditions([cond_1, cond_2, cond_3, cond_4])
        return config
