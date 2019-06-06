"""Controls imports to fill up dictionary of different sampling methods.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial


AL_MAPPING = {}



def get_base_AL_mapping():
    from rasa_nlu_gao.active_learning.sampling_methods.margin_AL import MarginAL
    from rasa_nlu_gao.active_learning.sampling_methods.informative_diverse import InformativeClusterDiverseSampler
    from rasa_nlu_gao.active_learning.sampling_methods.hierarchical_clustering_AL import HierarchicalClusterAL
    from rasa_nlu_gao.active_learning.sampling_methods.uniform_sampling import UniformSampling
    #from rasa_nlu_gao.active_learning.sampling_methods.represent_cluster_centers import RepresentativeClusterMeanSampling
    from rasa_nlu_gao.active_learning.sampling_methods.graph_density import GraphDensitySampler
    from rasa_nlu_gao.active_learning.sampling_methods.kcenter_greedy import kCenterGreedy
    AL_MAPPING['margin'] = MarginAL
    AL_MAPPING['informative_diverse'] = InformativeClusterDiverseSampler
    AL_MAPPING['hierarchical'] = HierarchicalClusterAL
    AL_MAPPING['uniform'] = UniformSampling
    #AL_MAPPING['margin_cluster_mean'] = RepresentativeClusterMeanSampling
    AL_MAPPING['graph_density'] = GraphDensitySampler
    AL_MAPPING['kcenter'] = kCenterGreedy


def get_all_possible_arms():
    from rasa_nlu_gao.active_learning.sampling_methods.mixture_of_samplers import MixtureOfSamplers
    AL_MAPPING['mixture_of_samplers'] = MixtureOfSamplers


def get_wrapper_AL_mapping():
    from rasa_nlu_gao.active_learning.sampling_methods.bandit_discrete import BanditDiscreteSampler
    from rasa_nlu_gao.active_learning.sampling_methods.simulate_batch import SimulateBatchSampler
    AL_MAPPING['bandit_mixture'] = partial(
        BanditDiscreteSampler,
        samplers=[{
            'methods': ['margin', 'uniform'],
            'weights': [0, 1]
        }, {
            'methods': ['margin', 'uniform'],
            'weights': [0.25, 0.75]
        }, {
            'methods': ['margin', 'uniform'],
            'weights': [0.5, 0.5]
        }, {
            'methods': ['margin', 'uniform'],
            'weights': [0.75, 0.25]
        }, {
            'methods': ['margin', 'uniform'],
            'weights': [1, 0]
        }])
    AL_MAPPING['bandit_discrete'] = partial(
        BanditDiscreteSampler,
        samplers=[{
            'methods': ['margin', 'uniform'],
            'weights': [0, 1]
        }, {
            'methods': ['margin', 'uniform'],
            'weights': [1, 0]
        }])
    AL_MAPPING['simulate_batch_mixture'] = partial(
        SimulateBatchSampler,
        samplers=({
            'methods': ['margin', 'uniform'],
            'weights': [1, 0]
        }, {
            'methods': ['margin', 'uniform'],
            'weights': [0.5, 0.5]
        }, {
            'methods': ['margin', 'uniform'],
            'weights': [0, 1]
        }),
        n_sims=5,
        train_per_sim=10,
        return_best_sim=False)
    AL_MAPPING['simulate_batch_best_sim'] = partial(
        SimulateBatchSampler,
        samplers=[{
            'methods': ['margin', 'uniform'],
            'weights': [1, 0]
        }],
        n_sims=10,
        train_per_sim=10,
        return_type='best_sim')
    AL_MAPPING['simulate_batch_frequency'] = partial(
        SimulateBatchSampler,
        samplers=[{
            'methods': ['margin', 'uniform'],
            'weights': [1, 0]
        }],
        n_sims=10,
        train_per_sim=10,
        return_type='frequency')

def get_mixture_of_samplers(name):
    assert 'mixture_of_samplers' in name
    if 'mixture_of_samplers' not in self.AL_MAPPING:
      raise KeyError('Mixture of Samplers not yet loaded.')
    args = name.split('-')[1:]
    samplers = args[0::2]
    weights = args[1::2]
    weights = [float(w) for w in weights]
    assert sum(weights) == 1
    mixture = {'methods': samplers, 'weights': weights}
    print(mixture)
    return partial(AL_MAPPING['mixture_of_samplers'], mixture=mixture)


def get_AL_sampler(name):
    if name in AL_MAPPING and name != 'mixture_of_samplers':
      return AL_MAPPING[name]
    if 'mixture_of_samplers' in name:
      return get_mixture_of_samplers(name)
    raise NotImplementedError('The specified sampler is not available.')
