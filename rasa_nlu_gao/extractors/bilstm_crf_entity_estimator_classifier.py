from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn
import logging
import re
import numpy as np


import typing


try:
    import cPickle as pickle
except ImportError:
    import pickle



from rasa_nlu_gao.extractors import EntityExtractor


from rasa_nlu_gao.utils.bilstm_utils import char_mapping, tag_mapping, prepare_dataset,prepare_dataset_for_estimator, iob_iobes, iob2
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import tensorflow as tf

try:
    import tensorflow as tf
except ImportError:
    tf = None


class BilstmCRFEntityEstimatorExtractor(EntityExtractor):
    name = "ner_bilstm_crf_estimator"

    provides = ["entities"]

    requires = ["tokens"]

    defaults = {
        "lr": 0.001,
        "char_dim": 100,
        "lstm_dim": 100,
        "batches_per_epoch": 10,
        "seg_dim": 20,
        "num_segs": 4,
        "batch_size": 10,
        "zeros": True,
        "tag_schema": "iobes",
        "lower": False,
        "model_type": "bilstm",
        "clip": 5,
        "optimizer": "adam",
        "dropout_keep": 0.5,
        "steps_check": 100,
        "config_proto": {
            "device_count": cpu_count(),
            "inter_op_parallelism_threads": 0,
            "intra_op_parallelism_threads": 0,
            "allow_growth": True,
            "allocator_type": 'BFC',  # best-fit with coalescing algorithm 内存分配、释放、碎片管理
            "per_process_gpu_memory_fraction": 0.25
        }
    }

    def __init__(self,
                 component_config=None,
                 ent_tagger=None,
                 session=None,
                 char_to_id=None,
                 id_to_tag=None):
        super(BilstmCRFEntityEstimatorExtractor, self).__init__(component_config)

        self.component_config = component_config
        self.ent_tagger = ent_tagger  # 指的是训练好的model
        self.session = session
        self.char_to_id = char_to_id
        self.id_to_tag = id_to_tag

    def train(self, training_data, config, **kwargs):
        self.component_config = config.for_component(self.name, self.defaults)

        if training_data.entity_examples:
            filtered_entity_examples = self.filter_trainable_entities(training_data.training_examples)

            train_sentences = self._create_dataset(filtered_entity_examples)

            # 检测并维护数据集的tag标记
            self.update_tag_scheme(
                train_sentences, self.component_config["tag_schema"])

            _c, char_to_id, id_to_char = char_mapping(
                train_sentences, self.component_config["lower"])

            tag_to_id, id_to_tag = tag_mapping(train_sentences)

            self.char_to_id = char_to_id
            self.id_to_tag = id_to_tag

            self.component_config["num_chars"] = len(char_to_id)
            self.component_config["num_tags"] = len(tag_to_id)

            train_data = prepare_dataset_for_estimator(
                train_sentences, char_to_id, tag_to_id, self.component_config["lower"]
            )

            # define feature spec for input x parsing
            feature_names = ['ChatInputs','SegInputs','Dropout']
            chatInput_feature = tf.feature_column.numeric_column(key='ChatInputs',
                                                                 shape=[1, len(train_data["chars"][0])])
            segInputs_feature = tf.feature_column.numeric_column(key='SegInputs',
                                                                 shape=[1, len(train_data["segs"][0])])
            dropout_feature = tf.feature_column.numeric_column(key='Dropout')

            self.feature_columns = [
                chatInput_feature,
                segInputs_feature,
                dropout_feature
            ]

            # set gpu and tf graph confing
            tf.logging.set_verbosity(tf.logging.INFO)

            config_proto = self.get_config_proto(self.component_config)

            # 创建实体识别模型
            classifier = tf.estimator.Estimator(
                model_fn=self.model_fn,
                params=self.component_config)

            chatInput_array = train_data["chars"]
            segInputs_array = train_data["segs"]



            x_tensor_train = {'ChatInputs':chatInput_array,'SegInputs':segInputs_array,'Dropout':0.5}
            classifier.train(input_fn=lambda: self.input_fn(x_tensor_train,
                                                    train_data["tags"],
                                                    self.component_config["batch_size"],
                                                    shuffle_num=100,
                                                    mode=tf.estimator.ModeKeys.TRAIN),
                                                    max_steps=800
                             )



    def _create_dataset(self, examples):
        dataset = []
        for example in examples:
            entity_offsets = self._convert_example(example)
            dataset.append(self._predata(
                example.text, entity_offsets, self.component_config["zeros"]))
        return dataset

    @staticmethod
    def _convert_example(example):
        def convert_entity(entity):
            return entity["start"], entity["end"], entity["entity"]

        return [convert_entity(ent) for ent in example.get("entities", [])]


    @staticmethod
    def _predata(text, entity_offsets, zeros):
        value = 'O'
        bilou = [value for _ in text]
        # zero_digits函数的用途是将所有数字转化为0

        def zero_digits(s):
            return re.sub('\d', '0', s)

        text = zero_digits(text.rstrip()) if zeros else text.rstrip()

        cooked_data = []

        for (start, end, entity) in entity_offsets:
            if start is not None and end is not None:
                bilou[start] = 'B-' + entity
                for i in range(start+1, end):
                    bilou[i] = 'I-' + entity

        for index, achar in enumerate(text):
            if achar.strip():
                temp = []
                temp.append(achar)
                temp.append(bilou[index])

                cooked_data.append(temp)
            else:
                continue

        return cooked_data


    def update_tag_scheme(self, sentences, tag_scheme):
        for i, s in enumerate(sentences):
            tags = [w[1] for w in s]
            # Check that tags are given in the IOB format
            if not iob2(tags):
                s_str = '\n'.join(' '.join(w) for w in s)
                raise Exception('Sentences should be given in IOB format! ' +
                                'Please check sentence %i:\n%s' % (i, s_str))
            if tag_scheme == 'iob':
                # If format was IOB1, we convert to IOB2
                for word, new_tag in zip(s, tags):
                    word[1] = new_tag
            elif tag_scheme == 'iobes':
                new_tags = iob_iobes(tags)
                for word, new_tag in zip(s, new_tags):
                    word[1] = new_tag
            else:
                raise Exception('Unknown tagging scheme!')


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


    def input_fn(self,features, labels, batch_size, shuffle_num, mode):
        """
         build tf.data set for input pipeline

        :param features: type dict() , define input x structure for parsing
        :param labels: type np.array input label
        :param batch_size: type int number ,input batch_size
        :param shuffle_num: type int number , random select the data
        :param mode: type string ,tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
        :return: set() with type of (tf.data , and labels)
        """
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(shuffle_num).batch(batch_size).repeat(self.epochs)
        else:
            dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        data, labels = iterator.get_next()
        return data, labels

    def model_fn(self,features, labels, mode,params):
        self.lr = params["lr"]
        self.char_dim = params["char_dim"]
        self.lstm_dim = params["lstm_dim"]
        self.seg_dim = params["seg_dim"]
        self.num_tags = params["num_tags"]
        self.num_chars = params["num_chars"]  # 样本中总字数
        self.num_segs = 4 # 0,1,2,3
        self.is_training = params["is_training"]

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)

        self.initializer = initializers.xavier_initializer()

        # # add placeholders for the model

        self.char_inputs = features['char_inputs']
        self.seg_inputs = features['SegInputs']
        self.dropout = features['Dropout']

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        # embeddings for chinese character and segmentation representation
        # 根据 char_inputs 和 seg_inputs 初始化向量
        embedding = self.embedding_layer(
            self.char_inputs, self.seg_inputs, params)

        model_inputs = tf.nn.dropout(embedding, self.dropout)

        model_outputs = self.biLSTM_layer(
            model_inputs, self.lstm_dim, self.lengths)

        # logits for tags
        self.logits = self.project_layer_bilstm(model_outputs)

        # 预测
        predictions = {
            'classes': tf.argmax(input=self.logits, axis=1, name='classes'),
            'probabilities': tf.nn.softmax(self.logits, name='softmax_tensor')
        }

        # 评估方法
        accuracy, update_op = tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'], name='accuracy')
        batch_acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(labels, tf.int64), predictions['classes']), tf.float32))
        tf.summary.scalar('batch_acc', batch_acc)
        tf.summary.scalar('streaming_acc', update_op)


        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=self.logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # loss of the model
            self.loss = self.loss_layer(self.logits, self.lengths)

            with tf.variable_scope("optimizer"):
                optimizer = self.config["optimizer"]
                if optimizer == "sgd":
                    self.opt = tf.train.GradientDescentOptimizer(self.lr)
                elif optimizer == "adam":
                    self.opt = tf.train.AdamOptimizer(self.lr)
                elif optimizer == "adgrad":
                    self.opt = tf.train.AdagradOptimizer(self.lr)
                else:
                    raise KeyError

                # apply grad clip to avoid gradient explosion
                # 梯度裁剪防止梯度爆炸
                grads_vars = self.opt.compute_gradients(self.loss)
                capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                     for g, v in grads_vars]
                # 更新梯度（可以用移动均值更新梯度试试，然后重新跑下程序）
                self.train_op = self.opt.apply_gradients(
                    capped_grads_vars, self.global_step)

                return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op)

        eval_metric_ops = {
            'accuracy': (accuracy, update_op)
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        此处只嵌入了两个特征，不同场景下可以嵌入不同特征，如果嵌入拼音特征、符号特征，应该可以用来检测错别字吧 0.0
        """
        #高:3 血:22 糖:23 和:24 高:3 血:22 压:25 char_inputs=[3,22,23,24,3,22,25]
        #高血糖和高血压 高血糖=[1,2,3] 和=[0] 高血压=[1,2,3]  seg_inputs=[1,2,3,0,1,2,3]
        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                name="char_embedding",
                shape=[self.num_chars, self.char_dim],
                initializer=self.initializer)
            # embedding_lookup详解：https://blog.csdn.net/yinruiyang94/article/details/77600453
            # 输入char_inputs='常' 对应的字典的索引/编号/value为：8
            # self.char_lookup=[2677*100]的向量，char_inputs字对应在字典的索引/编号/key=[1]
            embedding.append(tf.nn.embedding_lookup(
                self.char_lookup, char_inputs))
            #self.embedding1.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],  # shape=[4,20]
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(
                        self.seg_lookup, seg_inputs))
            # shape(?, ?, 120) 100维的字向量，20维的tag向量
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                model_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)


    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])