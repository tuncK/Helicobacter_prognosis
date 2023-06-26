# Worker script that defines the problem to be optimised by BOHB.
# From: https://automl.github.io/HpBandSter/build/html/index.html

import ConfigSpace as CS
from hpbandster.core.worker import Worker

import autoencoders


def extract_convert_params(config_dict):
    """
    A utility function to convert the hyperparams used for the optimisation.
    It is more efficient to log-sample some of the search dims, which needs to be converted.

    Args:
        config_dict: dictionary containing the sampled configurations by the optimizer

    Returns:
        tuple with the following fields:
            'grad_threshold' (scalar): Threshold on the magnitude of the gradient to
                allow during training.
            'dim' (integer): Number of dimensions in the latent layer
            'seed' (integer): Seed for RNG
        """

    grad_threshold = 10**config_dict['log10_grad_thres']
    dim = 2**config_dict['log2_latent_dim']
    seed = config_dict['seed']
    AE_type = config_dict['AE_type']
    return (grad_threshold, dim, seed, AE_type)


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

    def __init__(self, data_table, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_table = data_table

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

        (grad_threshold, dim, seed, AE_type) = extract_convert_params(config)
        val_loss = autoencoders.train_modality(Xfile=self.data_table, Yfile=None, gradient_threshold=grad_threshold,
                                               latent_dims=dim, max_training_duration=budget, seed=seed,
                                               AE_type=AE_type)
        return ({
                 'loss': float(val_loss),  # this is the a mandatory field to run hyperband
                 'info': val_loss  # can be used for any user-defined information - also mandatory
                })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('log10_grad_thres', lower=-6, upper=1))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('log2_latent_dim', lower=2, upper=10))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('seed', lower=1, upper=10))
        config_space.add_hyperparameter(CS.Categorical('AE_type', ['AE']))
        return (config_space)
