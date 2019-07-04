from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import io
import logging
import tempfile
import shutil

import datetime
import os
from builtins import object
from concurrent.futures import ProcessPoolExecutor as ProcessPool
from future.utils import PY3
from rasa_nlu_gao.training_data import Message

from rasa_nlu_gao import utils, config
from rasa_nlu_gao.components import ComponentBuilder
from rasa_nlu_gao.config import RasaNLUModelConfig
from rasa_nlu_gao.evaluate import get_evaluation_metrics, clean_intent_labels
from rasa_nlu_gao.model import InvalidProjectError,DeleteFailError
from rasa_nlu_gao.project import Project
from rasa_nlu_gao.train import do_train_in_worker, TrainingException
from rasa_nlu_gao.training_data.loading import load_data
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.logger import jsonFileLogObserver, Logger
from typing import Text, Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# in some execution environments `reactor.callFromThread`
# can not be called as it will result in a deadlock as
# the `callFromThread` queues the function to be called
# by the reactor which only happens after the call to `yield`.
# Unfortunately, the test is blocked there because `app.flush()`
# needs to be called to allow the fake server to
# respond and change the status of the Deferred on which
# the client is yielding. Solution: during tests we will set
# this Flag to `False` to directly run the calls instead
# of wrapping them in `callFromThread`.
DEFERRED_RUN_IN_REACTOR_THREAD = True


class MaxTrainingError(Exception):
    """Raised when a training is requested and the server has
        reached the max count of training processes.

    Attributes:
        message -- explanation of why the request is invalid
    """

    def __init__(self):
        self.message = 'The server can\'t train more models right now!'

    def __str__(self):
        return self.message


def deferred_from_future(future):
    """Converts a concurrent.futures.Future object to a
       twisted.internet.defer.Deferred object.

    See:
    https://twistedmatrix.com/pipermail/twisted-python/2011-January/023296.html
    """

    d = Deferred()

    def callback(future):
        e = future.exception()
        if e:
            if DEFERRED_RUN_IN_REACTOR_THREAD:
                reactor.callFromThread(d.errback, e)
            else:
                d.errback(e)
        else:
            if DEFERRED_RUN_IN_REACTOR_THREAD:
                reactor.callFromThread(d.callback, future.result())
            else:
                d.callback(future.result())

    future.add_done_callback(callback)
    return d


class DataRouter(object):
    def __init__(self,
                 project_dir=None,
                 max_training_processes=1,
                 response_log=None,
                 emulation_mode=None,
                 remote_storage=None,
                 component_builder=None):
        self._training_processes = max(max_training_processes, 1)
        self._current_training_processes = 0
        self.responses = self._create_query_logger(response_log)
        self.project_dir = config.make_path_absolute(project_dir)
        self.emulator = self._create_emulator(emulation_mode)
        self.remote_storage = remote_storage

        if component_builder:
            self.component_builder = component_builder
        else:
            self.component_builder = ComponentBuilder(use_cache=True)

        self.project_store = self._create_project_store(project_dir)
        self.pool = ProcessPool(self._training_processes)

    def __del__(self):
        """Terminates workers pool processes"""
        self.pool.shutdown()

    @staticmethod
    def _create_query_logger(response_log):
        """Create a logger that will persist incoming query results."""

        # Ensures different log files for different
        # processes in multi worker mode
        if response_log:
            # We need to generate a unique file name,
            # even in multiprocess environments
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file_name = "rasa_nlu_log-{}-{}.log".format(timestamp,
                                                            os.getpid())
            response_logfile = os.path.join(response_log, log_file_name)
            # Instantiate a standard python logger,
            # which we are going to use to log requests
            utils.create_dir_for_file(response_logfile)
            out_file = io.open(response_logfile, 'a', encoding='utf8')
            query_logger = Logger(
                    observer=jsonFileLogObserver(out_file, recordSeparator=''),
                    namespace='query-logger')
            # Prevents queries getting logged with parent logger
            # --> might log them to stdout
            logger.info("Logging requests to '{}'.".format(response_logfile))
            return query_logger
        else:
            # If the user didn't provide a logging directory, we wont log!
            logger.info("Logging of requests is disabled. "
                        "(No 'request_log' directory configured)")
            return None

    def _collect_projects(self, project_dir):
        projects = []
        if project_dir and os.path.isdir(project_dir):
            projects.extend(os.listdir(project_dir))
        else:
            projects = []

        projects.extend(self._list_projects_in_cloud())
        return projects

    def _create_project_store(self, project_dir):
        projects = self._collect_projects(project_dir)

        project_store = {}
        bot_model_dict = {}

        for project in projects:
            print(os.path.join(project_dir,project))
            for botId in os.listdir(os.path.join(project_dir,project)):
                bot_model_dict[botId]=Project(self.component_builder,
                                                 project_dir=self.project_dir,
                                                 tenanetId=project,
                                                 botId=botId,
                                                 remote_storage=self.remote_storage)
            project_store[project] = bot_model_dict

        if not project_store:
            default_model = RasaNLUModelConfig.DEFAULT_PROJECT_NAME
            project_store[default_model] = Project(
                    tenanetId=RasaNLUModelConfig.DEFAULT_TENANET_ID,
                    botId=RasaNLUModelConfig.DEFAULT_BOT_ID,
                    project_dir=self.project_dir,
                    remote_storage=self.remote_storage)
        return project_store

    def _pre_load(self, projects):
        logger.debug("loading %s", projects)
        for project in self.project_store:
            if project in projects:
                self.project_store[project].load_model()

    def _list_projects_in_cloud(self):
        try:
            from rasa_nlu_gao.persistor import get_persistor
            p = get_persistor(self.remote_storage)
            if p is not None:
                return p.list_projects()
            else:
                return []
        except Exception:
            logger.exception("Failed to list projects. Make sure you have "
                             "correctly configured your cloud storage "
                             "settings.")
            return []

    @staticmethod
    def _create_emulator(mode):
        """Create emulator for specified mode.

        If no emulator is specified, we will use the Rasa NLU format."""

        if mode is None:
            from rasa_nlu_gao.emulators import NoEmulator
            return NoEmulator()
        elif mode.lower() == 'wit':
            from rasa_nlu_gao.emulators.wit import WitEmulator
            return WitEmulator()
        elif mode.lower() == 'luis':
            from rasa_nlu_gao.emulators.luis import LUISEmulator
            return LUISEmulator()
        elif mode.lower() == 'dialogflow':
            from rasa_nlu_gao.emulators.dialogflow import DialogflowEmulator
            return DialogflowEmulator()
        else:
            raise ValueError("unknown mode : {0}".format(mode))

    @staticmethod
    def _tf_in_pipeline(model_config):
        # type: (RasaNLUModelConfig) -> bool
        from rasa_nlu_gao.classifiers.embedding_intent_classifier import \
            EmbeddingIntentClassifier
        return EmbeddingIntentClassifier.name in model_config.component_names

    def extract(self, data):
        return self.emulator.normalise_request_json(data)

    def parse(self, data):
        tenanetId = data.get("TenanetID", RasaNLUModelConfig.DEFAULT_PROJECT_NAME)
        botId = data.get("BotRecordId",RasaNLUModelConfig.DEFAULT_PROJECT_NAME)

        if tenanetId is None or botId is None:
            raise InvalidProjectError(
                "Can find model properly when tenanetId or botId is null ")


        if tenanetId not in self.project_store:
            projects = self._list_projects(self.project_dir)

            cloud_provided_projects = self._list_projects_in_cloud()
            projects.extend(cloud_provided_projects)

            if tenanetId not in projects:
                raise InvalidProjectError(
                        "No project found with tenanetId  '{}'.".format(tenanetId))
            else:
                try:
                    tempdict={}
                    tempdict[botId] =Project(
                            self.component_builder,
                            tenanetId=tenanetId,
                            botId=botId,
                            project_dir = self.project_dir)
                    self.project_store[tenanetId] = tempdict
                except Exception as e:
                    raise InvalidProjectError(
                            "Unable to load project tenanetId '{}'. "
                            "Error: {}".format(tenanetId, e))

        time = data.get('time')
        try:
            response = self.project_store[tenanetId][botId].parse(data['text'], time, requested_model_name=botId)
            logger.exception("tenanetId is {},botId is {} parsing data text : {}".format(tenanetId,botId,data['text']))
        except Exception as e:
            print(e)
            raise InvalidProjectError(
                "No project found with botId  '{}'.".format(botId))


        if self.responses:
            self.responses.info('', user_input=response,
                                model=response.get('model'))

        return self.format_response(response)

    @staticmethod
    def _list_projects(path):
        """List the projects in the path, ignoring hidden directories."""
        return [os.path.basename(fn)
                for fn in utils.list_subdirectories(path)]

    def parse_training_examples(self, examples, project, model):
        # type: (Optional[List[Message]], Text, Text) -> List[Dict[Text, Text]]
        """Parses a list of training examples to the project interpreter"""

        predictions = []
        for ex in examples:
            logger.debug("Going to parse: {}".format(ex.as_dict()))
            response = self.project_store[project].parse(ex.text,
                                                         None,
                                                         model)
            logger.debug("Received response: {}".format(response))
            predictions.append(response)

        return predictions

    def format_response(self, data):
        return self.emulator.normalise_response_json(data)

    def get_status(self):
        # This will only count the trainings started from this
        # process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.

        return {
            "max_training_processes": self._training_processes,
            "current_training_processes": self._current_training_processes,
            "available_projects": {
                name: proj[name].as_dict()
                for name, proj in self.project_store.items()
            }
        }

    def start_train_process(self,
                            data_file,  # type: Text
                            tenanetId,  # type: Text
                            train_config,  # type: RasaNLUModelConfig
                            botId=None  # type: Optional[Text]
                            ):
        # type: (...) -> Deferred
        """Start a model training."""

        if not tenanetId and not botId:
            raise InvalidProjectError("Missing tenanetId or botId to train")

        tempdict = {}

        if tenanetId in self.project_store:
            if self._training_processes <= self._current_training_processes:
                raise MaxTrainingError
            if botId in self.project_store[tenanetId]:
                tempdict = self.project_store[tenanetId][botId]
                tempdict.status = 1
            else:
                model = Project(
                    self.component_builder, tenanetId=tenanetId, botId=botId,
                    project_dir=self.project_dir, remote_storage=self.remote_storage)
                model.status = 1
                self.project_store[tenanetId][botId]=model
        elif tenanetId not in self.project_store:
            try:
                tempdict[botId] = Project(
                    self.component_builder, tenanetId=tenanetId, botId=botId,
                    project_dir=self.project_dir, remote_storage=self.remote_storage)
                tempdict[botId].status = 1
                self.project_store[tenanetId] = tempdict
            except Exception as e:
                print(e)


        def training_callback(model_path):
            model_dir = os.path.basename(os.path.normpath(model_path))
            self.project_store[tenanetId][botId].update(model_dir)
            self._current_training_processes -= 1
            self.project_store[tenanetId][botId].current_training_processes -= 1
            if (self.project_store[tenanetId][botId].status == 1 and
                    self.project_store[tenanetId][botId].current_training_processes ==
                    0):
                self.project_store[tenanetId][botId].status = 0
            return model_dir

        def training_errback(failure):
            logger.warning(failure)
            target_project = self.project_store.get(
                    failure.value.failed_target_project)
            self._current_training_processes -= 1
            self.project_store[tenanetId][botId].current_training_processes -= 1
            if (target_project and
                    self.project_store[tenanetId][botId].current_training_processes ==
                    0):
                target_project[botId].status = 0
            return failure

        logger.debug("New training queued")

        self._current_training_processes += 1
        self.project_store[tenanetId][botId].current_training_processes += 1

        # tensorflow training is not executed in a separate thread, as this may
        # cause training to freeze
        if self._tf_in_pipeline(train_config):
            try:
                logger.warning("Training a pipeline with a tensorflow "
                               "component. This blocks the server during "
                               "training.")
                model_path = do_train_in_worker(
                    train_config,
                    data_file,
                    path=self.project_dir,
                    tenanetId=tenanetId,
                    botId=botId,
                    storage=self.remote_storage)
                model_dir = os.path.basename(os.path.normpath(model_path))
                training_callback(model_dir)
                return model_dir
            except TrainingException as e:
                logger.warning(e)
                target_project = self.project_store.get(
                    e.failed_target_project)
                if target_project:
                    target_project.status = 0
                raise e
        else:
            result = self.pool.submit(do_train_in_worker,
                                      train_config,
                                      data_file,
                                      path=self.project_dir,
                                      tenanetId=tenanetId,
                                      botId=botId,
                                      storage=self.remote_storage)
            result = deferred_from_future(result)
            result.addCallback(training_callback)
            result.addErrback(training_errback)

            return result

    def evaluate(self, data, project=None, model=None):
        # type: (Text, Optional[Text], Optional[Text]) -> Dict[Text, Any]
        """Perform a model evaluation."""

        project = project or RasaNLUModelConfig.DEFAULT_PROJECT_NAME
        model = model or None
        file_name = utils.create_temporary_file(data, "_training_data")
        test_data = load_data(file_name)

        if project not in self.project_store:
            raise InvalidProjectError("Project {} could not "
                                      "be found".format(project))

        preds_json = self.parse_training_examples(test_data.intent_examples,
                                                  project,
                                                  model)

        predictions = [
            {"text": e.text,
             "intent": e.data.get("intent"),
             "predicted": p.get("intent", {}).get("name"),
             "confidence": p.get("intent", {}).get("confidence")}
            for e, p in zip(test_data.intent_examples, preds_json)
        ]

        y_true = [e.data.get("intent") for e in test_data.intent_examples]
        y_true = clean_intent_labels(y_true)

        y_pred = [p.get("intent", {}).get("name") for p in preds_json]
        y_pred = clean_intent_labels(y_pred)

        report, precision, f1, accuracy = get_evaluation_metrics(y_true,
                                                                 y_pred)

        return {
            "intent_evaluation": {
                "report": report,
                "predictions": predictions,
                "precision": precision,
                "f1_score": f1,
                "accuracy": accuracy}
        }



    def delete(self,path,tenanetId, botId):
        # type: (Text, Text)
        """Perform a model evaluation."""

        model_name = ""

        path = config.make_path_absolute(path)
        up_dir_name = os.path.join(path, tenanetId)
        dir_name = os.path.join(path, tenanetId,botId, model_name)

        try:

            if os.path.exists(dir_name):
                for f in os.listdir(dir_name):
                    filepath = os.path.join(dir_name, f)
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath, True)
                shutil.rmtree(dir_name, True)

            if os.path.exists(up_dir_name):
                shutil.rmtree(up_dir_name,True)
            return {
                        "Code": "1",
                        "TenanetID": tenanetId,
                        "Message": "",
                        "BotRecordId": botId
                    }
        except DeleteFailError as e:
            raise e


    def unload_model(self, tenanetId,botId):
        # type: (Text, Text) -> Dict[Text]
        """Unload a model from server memory."""
        if tenanetId is None:
            raise InvalidProjectError("Can't find model when tenanetId is none")
        elif tenanetId not in self.project_store:
            raise InvalidProjectError("Can't find model with tenanetId {}".format(tenanetId))
        try:
            unloaded_model = self.project_store[tenanetId][botId].unload(botId)
            return unloaded_model
        except KeyError:
            raise InvalidProjectError("Failed to unload model {} ".format(botId))
