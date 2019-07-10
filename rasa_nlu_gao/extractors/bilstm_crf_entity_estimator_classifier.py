from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn
import logging
import re
import os
import io
import functools
import time
import numpy as np
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode


import typing


try:
    import cPickle as pickle
except ImportError:
    import pickle



from rasa_nlu_gao.extractors import EntityExtractor


from rasa_nlu_gao.utils.bilstm_utils import char_mapping, tag_mapping, prepare_dataset_for_estimator, iob_iobes, iob2, input_from_line_for_estimator, result_to_json
from multiprocessing import cpu_count
from tensorflow.contrib import predictor as Pred

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
        "dropout": 0.5,
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
                 char_to_id=None,
                 id_to_tag=None,
                 predictor=None):
        super(BilstmCRFEntityEstimatorExtractor, self).__init__(component_config)

        self.component_config = component_config
        self.char_to_id = char_to_id
        self.id_to_tag = id_to_tag
        self.predictor = predictor

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
            self.component_config["is_training"] = True

            train_data = prepare_dataset_for_estimator(
                train_sentences, char_to_id, tag_to_id, self.component_config["lower"]
            )
            # set gpu and tf graph confing
            tf.logging.set_verbosity(tf.logging.INFO)

            # 创建实体识别模型
            self.estimator = tf.estimator.Estimator(
                model_fn=self.model_fn,
                params=self.component_config)

            chatInput_array = train_data["chars"]
            segInputs_array = train_data["segs"]
            self.max_length = train_data["max_length"]
            self.component_config['max_length']=self.max_length

            x_tensor_train = (chatInput_array,segInputs_array)
            self.estimator.train(input_fn=lambda: self.input_fn(x_tensor_train,
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

    def generator_fn(self,features, labels):
        char_inputs, seg_inputs = features
        for i in range(len(char_inputs)):
            try:
                yield (char_inputs[i], seg_inputs[i]), labels[i]
            except Exception as e:
                print(str(e))




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
        shapes = (([None],[None]), [None])
        types = ((tf.int32,tf.int32), tf.int32)

        dataset = tf.data.Dataset.from_generator(
            functools.partial(self.generator_fn, features, labels),
            output_shapes=shapes, output_types=types)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(shuffle_num).padded_batch(batch_size, shapes).repeat()
        else:
            dataset = dataset.padded_batch(batch_size, shapes)
        iterator = dataset.make_one_shot_iterator()
        data, labels = iterator.get_next()
        return data, labels

    def model_fn(self,features, labels, mode, params):
        self.lr = params["lr"]
        self.char_dim = params["char_dim"]
        self.lstm_dim = params["lstm_dim"]
        self.seg_dim = params["seg_dim"]
        self.num_tags = params["num_tags"]
        self.num_chars = params["num_chars"]  # 样本中总字数
        self.num_segs = 4 # 0,1,2,3
        self.is_training = params["is_training"]


        self.initializer = initializers.xavier_initializer()
        if mode == tf.estimator.ModeKeys.PREDICT:
            self.char_inputs = features['IteratorGetNext:0']
            self.seg_inputs = features['IteratorGetNext:1']
        else:
            self.char_inputs, self.seg_inputs = features


        self.dropout = params["dropout"]

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        # embeddings for chinese character and segmentation representation
        # 根据 char_inputs 和 seg_inputs 初始化向量
        embedding = self.embedding_layer(
            self.char_inputs, self.seg_inputs, params)
        if mode == tf.estimator.ModeKeys.PREDICT:
            model_inputs = embedding
        else:
            model_inputs = tf.nn.dropout(embedding, self.dropout)
        model_outputs = self.biLSTM_layer(
            model_inputs, self.lstm_dim, self.lengths)
        # logits for tags
        self.logits = self.project_layer_bilstm(model_outputs)
        if mode == tf.estimator.ModeKeys.PREDICT:
            self.targets = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="Targets")
        else:
            self.targets = labels
        self.loss = self.loss_layer(self.logits, self.lengths)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    'logits': self.logits,
                    'length': self.lengths
                })

        else:
            with tf.variable_scope("optimizer"):
                optimizer = params["optimizer"]
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
                capped_grads_vars = [[tf.clip_by_value(g, -params["clip"], params["clip"]), v]
                                     for g, v in grads_vars]
                # 更新梯度（可以用移动均值更新梯度试试，然后重新跑下程序）
                self.train_op = self.opt.apply_gradients(
                    capped_grads_vars, tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op)

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            # start_logits.shape (?, 1, 52)
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(
                small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            # project_logits.shape (?, ?, 51)
            # pad_logits.shape (?, ?, 1)
            # logits.shape (?, ?, 52)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)

            # crf_log_likelihood在一个条件随机场里面计算标签序列的log-likelihood
            # inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,
            # 一般使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入.
            # tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签.
            # sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度.
            # transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            # log_likelihood: 标量, log-likelihood
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)


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





    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        if self.estimator is None:
            return {"classifier_file": None}
        model_dir = os.path.join(model_dir)
        if 1 - os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.feature_columns = (
            tf.feature_column.numeric_column(key='IteratorGetNext:0', shape=[self.max_length],dtype=tf.int64),
            tf.feature_column.numeric_column(key='IteratorGetNext:1', shape=[self.max_length],dtype=tf.int64)
        )

        # build feature spec for tf.example parsing
        feature_spec = tf.feature_column.make_parse_example_spec(self.feature_columns)
        # build tf.example parser
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        path = self.estimator.export_saved_model(model_dir, serving_input_receiver_fn)
        # decode model path to string
        file_dir = os.path.basename(path).decode('utf-8')

        with io.open(os.path.join(
                model_dir,
                self.name + "_char_to_id.pkl"), 'wb') as f:
            pickle.dump(self.char_to_id, f)
        with io.open(os.path.join(
                model_dir,
                self.name + "_id_to_tag.pkl"), 'wb') as f:
            pickle.dump(self.id_to_tag, f)
        return {"classifier_file": file_dir}

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> EmbeddingBertIntentAdanetClassifier

        meta = model_metadata.for_component(cls.name)
        meta.update({'dropout': 0.0})
        config_proto = cls.get_config_proto(meta)

        logger.info("bert model loaded")

        if model_dir and meta.get("classifier_file"):
            file_name = meta.get("classifier_file")
            # tensorflow.contrib.predictor to load the model file which may has 10x speed up in predict time
            predict = Pred.from_saved_model(export_dir=os.path.join(model_dir, file_name), config=config_proto)

            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_char_to_id.pkl"), 'rb') as f:
                char_to_id = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_id_to_tag.pkl"), 'rb') as f:
                id_to_tag = pickle.load(f)

            return BilstmCRFEntityEstimatorExtractor(
                component_config=meta,
                char_to_id=char_to_id,
                id_to_tag=id_to_tag,
                predictor=predict
            )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return BilstmCRFEntityEstimatorExtractor(component_config=meta)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        start = time.time()
        extracted = self.add_extractor_name(self.extract_line(message))
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)
        end = time.time()
        logger.info("bilstm estimator entity extraction time cost %.3f s" % (end - start))

    def extract_line(self,message):
        result = self.extract_entities(message)
        return result.get("entities", [])

    def extract_entities(self, message):
        # type: (Message) -> List[Dict[Text, Any]]
        """Take a sentence and return entities in json format"""
        if self.predictor is not None:
            start_interal = time.time()
            inputs = input_from_line_for_estimator(message.text, self.char_to_id)
            char_input_list = inputs['char_input']
            char_input_array = np.pad(np.array(char_input_list), ((0,self.component_config['max_length']-len(char_input_list))), 'constant')
            char_input_list = char_input_array.tolist()
            seg_input_list = inputs['seg_input']
            seg_input_array = np.pad(np.array(seg_input_list), ((0,self.component_config['max_length']-len(seg_input_list))), 'constant')
            seg_input_list = seg_input_array.tolist()

            examples = []
            feature = {}
            # convert input x to tf.feature with float feature spec
            feature['IteratorGetNext:0'] = tf.train.Feature(int64_list=tf.train.Int64List(value=char_input_list))
            feature['IteratorGetNext:1'] = tf.train.Feature(int64_list=tf.train.Int64List(value=seg_input_list))
            # build tf.example for prediction
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=feature
                )
            )
            examples.append(example.SerializeToString())

            # Make predictions.
            logger.info("estimator prediction finished")
            end_interal = time.time()
            logger.info("prepare for entity extracting time cost %.3f s" % (end_interal -start_interal))
            begin_predict_interal = time.time()
            result_dict = self.predictor({'examples': examples})
            end_predict_interal = time.time()
            logger.info("prediction in entity extracting time cost %.3f s" % (end_predict_interal - begin_predict_interal))

            begin_process_predict = time.time()
            sess = self.predictor.session
            graph = sess.graph
            variables = graph.get_collection("variables")
            var_transition = variables[len(variables)-1]
            transition = var_transition.eval(session=sess)
            batch_paths = self.decode(result_dict["logits"], result_dict["length"], transition)
            tags = [self.id_to_tag[idx] for idx in batch_paths[0]]
            end_process_predict = time.time()
            logger.info(
                "after prediction in entity extracting time cost %.3f s" % (
                            end_process_predict - begin_process_predict))

            return result_to_json(inputs['text'], tags)
        else:
            return []


    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.component_config["num_tags"] + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths