from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os

import typing
from typing import List, Text, Any, Optional, Dict


from rasa_nlu_gao.components import Component
from rasa_nlu_gao.active_learning.sampling_methods.constants import get_AL_sampler
from rasa_nlu_gao.active_learning.sampling_methods.constants import AL_MAPPING
from rasa_nlu_gao.active_learning.sampling_methods.constants import get_wrapper_AL_mapping
from rasa_nlu_gao.active_learning.utils import utils
from multiprocessing import cpu_count
import numpy as np
import keras
from keras.models import load_model
handler = logging.FileHandler("log.txt")

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import tensorflow as tf
    from rasa_nlu_gao.config import RasaNLUModelConfig
    from rasa_nlu_gao.training_data import TrainingData
    from rasa_nlu_gao.model import Metadata
    from rasa_nlu_gao.training_data import Message

    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import StandardScaler


try:
    import tensorflow as tf

except ImportError:
    tf = None

get_wrapper_AL_mapping()

class ActiveBertNerualClassifier(Component):
    """Intent classifier using supervised bert embeddings."""

    name = "active_bert_nerual_classifier"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    defaults = {
        # nn architecture
        "batch_size": 0.8,
        "epochs": 2500,
        "sampling_method":'informative_diverse',    # sampling_method for selecting data from training set
        "score_method": 'small_cnn',# evaluate method
        "seed":1,                     # random seed for number generating
        "warmstart_size":0.8,         # Float indicates percentage of training data to use in the initial warmstart model
        "select_model": "small_cnn",           # If select model is None then the select model is equal to score_method
        "c":0.0,                       # Percentage of labels to randomize
        "active_sampling_percentage":1.0,   # Mixture weights on active sampling.
        "max_dataset_size":15000,      # maximum number of datapoints to include in data zero indicates no limit
        "standardize_data":True,       # Whether to standardize the data.
        "normalize_data":False,        # Whether to normalize the data.
        "train_horizon":1.0,           # how far to extend learning curve as a percent of train



        # flag if tokenize intents
        "intent_tokenization_flag": False,
        "intent_split_symbol": '_',

        "config_proto": {
            "device_count": cpu_count(),
            "inter_op_parallelism_threads": 0,
            "intra_op_parallelism_threads": 0,
            "allow_growth": True,
            "allocator_type": 'BFC',               # best-fit with coalescing algorithm 内存分配、释放、碎片管理
            "per_process_gpu_memory_fraction": 0.5 # this means use 50% of your gpu memory in max
        }
    }

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["tensorflow"]

    def _load_nn_architecture_params(self):
        self.batch_size = self.component_config['batch_size']
        self.epochs = self.component_config['epochs']
        self.sampling_method = self.component_config['sampling_method']
        self.score_method = self.component_config['score_method']
        self.seed = self.component_config['seed']
        self.warmstart_size = self.component_config['warmstart_size']
        self.select_model = self.component_config['select_model']
        self.c = self.component_config['c']
        self.m = self.component_config['active_sampling_percentage']
        self.max_dataset_size = self.component_config['max_dataset_size']
        self.standardize_data = self.component_config['standardize_data']
        self.normalize_data = self.component_config['normalize_data']
        self.train_horizon = self.component_config['train_horizon']

    def _load_flag_if_tokenize_intents(self):
        self.intent_tokenization_flag = self.component_config['intent_tokenization_flag']
        self.intent_split_symbol = self.component_config['intent_split_symbol']
        if self.intent_tokenization_flag and not self.intent_split_symbol:
            logger.warning("intent_split_symbol was not specified, "
                           "so intent tokenization will be ignored")
            self.intent_tokenization_flag = False

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                'Failed to import `tensorflow`. '
                'Please install `tensorflow`. '
                'For example with `pip install tensorflow`.')

    def __init__(self,
                 component_config=None,  # type: Optional[Dict[Text, Any]]
                 inv_intent_dict=None,  # type: Optional[Dict[int, Text]]
                 encoded_all_intents=None,  # type: Optional[np.ndarray]
                 results=None,
                 score_model=None
                 ):
        # type: (...) -> None
        """Declare instant variables with default values"""
        self._check_tensorflow()
        super(ActiveBertNerualClassifier, self).__init__(component_config)

        # nn architecture parameters
        self._load_nn_architecture_params()

        # flag if tokenize intents
        self._load_flag_if_tokenize_intents()

        # transform numbers to intents
        self.inv_intent_dict = inv_intent_dict
        # encode all intents with numbers
        self.encoded_all_intents = encoded_all_intents

        self.results = results

        self.score_model = score_model




    # training data helpers:
    @staticmethod
    def _create_intent_dict(training_data):
        """Create intent dictionary"""

        distinct_intents = set([example.get("intent")
                               for example in training_data.intent_examples])
        return {intent: idx
                for idx, intent in enumerate(sorted(distinct_intents))}

    @staticmethod
    def _create_intent_token_dict(intents, intent_split_symbol):
        """Create intent token dictionary"""

        distinct_tokens = set([token
                               for intent in intents
                               for token in intent.split(
                                        intent_split_symbol)])
        return {token: idx
                for idx, token in enumerate(sorted(distinct_tokens))}

    def _create_encoded_intents(self, intent_dict):
        """Create matrix with intents encoded in rows as bag of words,
        if intent_tokenization_flag = False this is identity matrix"""

        if self.intent_tokenization_flag:
            intent_token_dict = self._create_intent_token_dict(
                list(intent_dict.keys()), self.intent_split_symbol)

            encoded_all_intents = np.zeros((len(intent_dict),
                                            len(intent_token_dict)))
            for key, idx in intent_dict.items():
                for t in key.split(self.intent_split_symbol):
                    encoded_all_intents[idx, intent_token_dict[t]] = 1

            return encoded_all_intents
        else:
            return np.eye(len(intent_dict))

    # data helpers:
    def _create_all_Y(self, size):
        # stack encoded_all_intents on top of each other
        # to create candidates for training examples
        # to calculate training accuracy
        all_Y = np.stack([self.encoded_all_intents[0] for _ in range(size)])

        return all_Y

    def _prepare_data_for_training(self, training_data, intent_dict):
        """Prepare data for training"""

        X = np.stack([e.get("text_features")
                      for e in training_data.intent_examples])

        intents_for_X = np.array([intent_dict[e.get("intent")]
                                  for e in training_data.intent_examples])

        Y = np.stack([self.encoded_all_intents[intent_idx]
                      for intent_idx in intents_for_X])

        return X, Y, intents_for_X




    def train(self, training_data, cfg=None, **kwargs):
        # type: (TrainingData, Optional[RasaNLUModelConfig], **Any) -> None
        """Train the embedding intent classifier on a data set."""

        intent_dict = self._create_intent_dict(training_data)

        if len(intent_dict) < 2:
            logger.error("Can not train an intent classifier. "
                         "Need at least 2 different classes. "
                         "Skipping training of intent classifier.")
            return

        self.inv_intent_dict = {v: k for k, v in intent_dict.items()}
        self.encoded_all_intents = self._create_encoded_intents(intent_dict)

        X, Y, intents_for_X = self._prepare_data_for_training(training_data, intent_dict)

        all_results = {}

        sampler = get_AL_sampler(self.sampling_method)


        score_model = utils.get_model(self.score_method, self.seed)

        select_model = utils.get_model(self.select_model, self.seed)

        max_dataset_size = None if self.max_dataset_size == "0" else int(
            self.max_dataset_size)

        standardize_data = self.standardize_data == "True"

        normalize_data = self.normalize_data == "True"


        results, sampler_state = self.generate_one_curve(
            X,intents_for_X, sampler, score_model, self.seed, self.warmstart_size,
            self.batch_size, select_model, self.c, self.m, max_dataset_size,
            standardize_data, normalize_data, self.train_horizon)

        key = (self.sampling_method, self.score_method,
               select_model, self.m, self.warmstart_size, self.batch_size,
               self.c, standardize_data, normalize_data, self.seed)
        sampler_output = sampler_state.to_dict()
        results["sampler_output"] = sampler_output
        all_results[key] = results
        self.results = results
        #self.results["score_model"].release()





    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if self.score_method is None:
            logger.error("There is no trained tf.session: "
                         "component is either not trained or "
                         "didn't receive enough training data")

        else:
            X = message.get("text_features").tolist()

            X = np.reshape(X, (1, len(X)))


            result_score_list = self.score_model.predict(X)
            max_score = np.max(result_score_list)
            max_index = np.argmax(result_score_list)

            # if X contains all zeros do not predict some label
            if len(X)>0:
                intent = {
                    "name": self.inv_intent_dict[max_index], "confidence": float(max_score)
                }
                ranking = result_score_list[:len(result_score_list)]
                intent_ranking = [{"name": self.inv_intent_dict[intent_idx],
                                   "confidence": float(score)}
                                  for intent_idx, score in enumerate(ranking[0])]

                intent_ranking = sorted(intent_ranking, key=lambda s: s['confidence'], reverse=True)

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self, model_dir):

        filename = "text_active_bert_classifier.h5"

        path = os.path.join(model_dir,filename)

        self.results["score_model"].model.save(path)

        with io.open(os.path.join(
                model_dir,
                self.name + "_inv_intent_dict.pkl"), 'wb') as f:
            pickle.dump(self.inv_intent_dict, f)
        with io.open(os.path.join(
                model_dir,
                self.name + "_encoded_all_intents.pkl"), 'wb') as f:
            pickle.dump(self.encoded_all_intents, f)

        return {"classifier_file": filename}


    @staticmethod
    def get_config_proto(component_config):
        # 配置configProto
        config = tf.ConfigProto(
            device_count={
                'CPU': component_config['config_proto']['device_count']
            },
            inter_op_parallelism_threads=component_config['config_proto']['inter_op_parallelism_threads'],
            intra_op_parallelism_threads=component_config['config_proto']['intra_op_parallelism_threads'],
            gpu_options={
                'allow_growth': component_config['config_proto']['allow_growth']
            }
        )
        config.gpu_options.per_process_gpu_memory_fraction= component_config['config_proto']['per_process_gpu_memory_fraction']
        config.gpu_options.allocator_type = component_config['config_proto']['allocator_type']
        return config

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> EmbeddingBertIntentAdanetClassifier

        meta = model_metadata.for_component(cls.name)
        config_proto = cls.get_config_proto(meta)

        print("bert model loaded")

        if model_dir and meta.get("classifier_file"):
            file_name = meta.get("classifier_file")

            sess = tf.Session(config=config_proto)
            keras.backend.set_session(sess)

            path = os.path.join(model_dir,file_name)
            model = load_model(path)

            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_inv_intent_dict.pkl"), 'rb') as f:
                inv_intent_dict = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_encoded_all_intents.pkl"), 'rb') as f:
                encoded_all_intents = pickle.load(f)

            return ActiveBertNerualClassifier(
                    component_config=meta,
                    inv_intent_dict=inv_intent_dict,
                    encoded_all_intents=encoded_all_intents,
                    score_model=model
            )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return ActiveBertNerualClassifier(component_config=meta)



    def generate_one_curve(self,
                           X,
                           y,
                           sampler,
                           score_model,
                           seed,
                           warmstart_size,
                           batch_size,
                           select_model=None,
                           confusion=0.,
                           active_p=1.0,
                           max_points=None,
                           standardize_data=False,
                           norm_data=False,
                           train_horizon=0.5):


        def select_batch(sampler, uniform_sampler, mixture, N, already_selected,
                         **kwargs):
            n_active = int(mixture * N)
            n_passive = N - n_active
            kwargs["N"] = n_active
            kwargs["already_selected"] = already_selected
            batch_AL = sampler.select_batch(**kwargs)
            already_selected = already_selected + batch_AL
            kwargs["N"] = n_passive
            kwargs["already_selected"] = already_selected
            batch_PL = uniform_sampler.select_batch(**kwargs)
            return batch_AL + batch_PL

        np.random.seed(seed)
        data_splits = [2. / 3, 1. / 6, 1. / 6]

        # 2/3 of data for training
        if max_points is None:
            max_points = len(y)
        train_size = int(min(max_points, len(y)) * data_splits[0])
        if batch_size < 1:
            batch_size = int(batch_size * train_size)
        else:
            batch_size = int(batch_size)
        if warmstart_size < 1:
            # Set seed batch to provide enough samples to get at least 4 per class
            # TODO(lishal): switch to sklearn stratified sampler
            seed_batch = int(warmstart_size * train_size)
        else:
            seed_batch = int(warmstart_size)
        seed_batch = max(seed_batch, 6 * len(np.unique(y)))

        indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise = (
            utils.get_train_val_test_splits(X, y, max_points, seed, confusion,
                                            seed_batch, split=data_splits))

        # Preprocess data
        if norm_data:
            print("Normalizing data")
            X_train = normalize(X_train)
            X_val = normalize(X_val)
            X_test = normalize(X_test)
        if standardize_data:
            print("Standardizing data")
            nsamples, nx, ny, size = X_train.shape
            X_train = X_train.reshape((nsamples, nx * ny * size))
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)

            print(X_val.shape)
            nval_samples, nval_x, nval_y, nval_size = X_val.shape
            X_val = X_val.reshape((nval_samples, nval_x * nval_y * nval_size))
            X_val = scaler.transform(X_val)

            print(X_test.shape)
            ntest_samples, ntest_x, ntest_y, ntest_size = X_test.shape
            X_test = X_test.reshape((ntest_samples, ntest_x * ntest_y * ntest_size))
            X_test = scaler.transform(X_test)
        print("active percentage: " + str(active_p) + " warmstart batch: " +
              str(seed_batch) + " batch size: " + str(batch_size) + " confusion: " +
              str(confusion) + " seed: " + str(seed))

        # Initialize samplers
        uniform_sampler = AL_MAPPING["uniform"](X_train, y_train, seed)
        sampler = sampler(X_train, y_train, seed)

        results = {}
        data_sizes = []
        accuracy = []
        selected_inds = range(seed_batch)

        # If select model is None, use score_model
        same_score_select = False
        if select_model is None:
            select_model = score_model
            same_score_select = True


        print("train_horizon :" + str(train_horizon))
        print("train_size :" + str(train_size))
        print("seed_batch :" + str(seed_batch))
        print("batch_size :" + str(batch_size))


        n_batches = int(np.ceil((train_horizon * train_size - seed_batch) *
                                1.0 / batch_size)) + 1

        print("Total trainng batches : " + str(n_batches))
        score_model.create_y_mat(y)
        select_model.create_y_mat(y)

        for b in range(n_batches):
            print(b)
            n_train = seed_batch + min(train_size - seed_batch, b * batch_size)
            print("Training model on " + str(n_train) + " datapoints")

            assert n_train == len(selected_inds)
            data_sizes.append(n_train)

            # Sort active_ind so that the end results matches that of uniform sampling
            partial_X = X_train[sorted(selected_inds)]
            partial_y = y_train[sorted(selected_inds)]
            score_model.fit(partial_X, partial_y)
            if not same_score_select:
                select_model.fit(partial_X, partial_y)
            acc = score_model.score(X_test, y_test)
            accuracy.append(acc)
            print("Sampler: %s, Accuracy: %.2f%%" % (sampler.name, accuracy[-1] * 100))

            n_sample = min(batch_size, train_size - len(selected_inds))
            select_batch_inputs = {
                "model": select_model,
                "labeled": dict(zip(selected_inds, y_train[selected_inds])),
                "eval_acc": accuracy[-1],
                "X_test": X_val,
                "y_test": y_val,
                "y": y_train
            }
            selected_inds = list(selected_inds)
            new_batch = select_batch(sampler, uniform_sampler, active_p, n_sample,
                                     selected_inds, **select_batch_inputs)
            selected_inds.extend(new_batch)
            print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
            assert len(new_batch) == n_sample
            assert len(list(set(selected_inds))) == len(selected_inds)

        # Check that the returned indice are correct and will allow mapping to
        # training set from original data
        assert all(y_noise[indices[selected_inds]] == y_train[selected_inds])
        results["accuracy"] = accuracy
        results["selected_inds"] = selected_inds
        results["data_sizes"] = data_sizes
        results["indices"] = indices
        results["noisy_targets"] = y_noise
        results["score_model"] = score_model

        return results, sampler





