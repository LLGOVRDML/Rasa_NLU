from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
import os
import warnings
import time
import copy
from builtins import str
from typing import Any
from typing import Dict
from typing import Optional
from typing import Text
logger = logging.getLogger(__name__)

from rasa_nlu_gao import utils
from rasa_nlu_gao.extractors import EntityExtractor
from rasa_nlu_gao.model import Metadata
from rasa_nlu_gao.training_data import Message
from rasa_nlu_gao.training_data import TrainingData
from rasa_nlu_gao.utils import write_json_to_file

ENTITY_SYNONYMS_FILE_NAME = "entity_synonyms.json"
ENTITY_LOOKUPTABLES_FILE_NAME = "entity_lookuptables.json"


class EntitySynonymMapper(EntityExtractor):
    name = "ner_synonyms"

    provides = ["entities"]

    def __init__(self, component_config=None, synonyms=None,lookup_tables=None):
        # type: (Optional[Dict[Text, Text]]) -> None

        super(EntitySynonymMapper, self).__init__(component_config)

        self.synonyms = synonyms if synonyms else {}

        self.lookup_tables = lookup_tables if lookup_tables else {}

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None
        self.lookup_tables = training_data.lookup_tables
        for key, value in list(training_data.entity_synonyms.items()):
            self.add_entities_if_synonyms(key, value)

        for example in training_data.entity_examples:
            for entity in example.get("entities", []):
                entity_val = example.text[entity["start"]:entity["end"]]

                self.add_entities_if_synonyms(entity_val,
                                              str(entity.get("value")))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        start = time.time()
        updated_entities = message.get("entities", [])[:]
        self.replace_synonyms(updated_entities)
        message.set("entities", updated_entities, add_to_output=True)
        end = time.time()
        logger.info("synonyms time cost %.3f s" % (end - start))

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]

        if self.synonyms:
            entity_synonyms_file = os.path.join(model_dir,
                                                ENTITY_SYNONYMS_FILE_NAME)
            write_json_to_file(entity_synonyms_file, self.synonyms,
                               separators=(',', ': '))

            entity_lookuptables_file = os.path.join(model_dir,
                                                ENTITY_LOOKUPTABLES_FILE_NAME)
            write_json_to_file(entity_lookuptables_file, self.lookup_tables,
                               separators=(',', ': '))

        return {"synonyms_file": ENTITY_SYNONYMS_FILE_NAME,"lookup_tables_file": ENTITY_LOOKUPTABLES_FILE_NAME}

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[EntitySynonymMapper]
             **kwargs  # type: **Any
             ):
        # type: (...) -> EntitySynonymMapper

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("synonyms_file", ENTITY_SYNONYMS_FILE_NAME)
        entity_synonyms_file = os.path.join(model_dir, file_name)

        if os.path.isfile(entity_synonyms_file):
            synonyms = utils.read_json_file(entity_synonyms_file)
        else:
            synonyms = None
            warnings.warn("Failed to load synonyms file from '{}'"
                          "".format(entity_synonyms_file))

        lookuptables_filename = meta.get("lookup_tables_file", ENTITY_LOOKUPTABLES_FILE_NAME)
        entity_lookuptables_file = os.path.join(model_dir, lookuptables_filename)

        if os.path.isfile(entity_lookuptables_file):
            lookup_tables = utils.read_json_file(entity_lookuptables_file)
        else:
            lookup_tables = None
            warnings.warn("Failed to load lookup_tables file from '{}'"
                          "".format(entity_lookuptables_file))

        return EntitySynonymMapper(meta, synonyms,lookup_tables)

    def replace_synonyms(self, entities):
        for entity in entities:
            # need to wrap in `str` to handle e.g. entity values of type int
            entity_value = str(entity["value"])
            if entity_value.lower() in self.synonyms:
                value_dict = {}
                value_list = []
                value_list.append(self.synonyms[entity_value.lower()])
                value_dict["values"] = value_list
                entity["resolution"] = value_dict
                self.add_processor_name(entity)
        returnEntities = []
        for entity in entities:
            if entity["resolution"]:
                resList = [item["resolution"] for item in self.lookup_tables]
                matchList=[]
                # if entity['resolution']['values'][0] not in resList:
                #     for item in self.lookup_tables:
                #         if entity["resolution"]['values'][0] in item["elements"]:
                #             matchList.append(item)
               
                #     # matchList = [item if entity["resolution"]['values'][0] in item["elements"] else None for item in self.lookup_tables]
                #     for matchItem in matchList:
                #         if matchItem:
                #             newEntity = copy.deepcopy(entity)
                #             newEntity['entity']=matchItem['name']
                #             newEntity['resolution']['values'][0]=matchItem['resolution']
                #             returnEntities.append(newEntity)
                # else:
                #     returnEntities.append(entity)
                for item in self.lookup_tables:
                    if entity['value'] in item["elements"]:
                        matchList.append(item)
                for matchItem in matchList:
                    if matchItem:
                        newEntity = copy.deepcopy(entity)
                        newEntity['entity']=matchItem['name']
                        newEntity['resolution']['values'][0]=matchItem['resolution']
                        returnEntities.append(newEntity)
            else:
                returnEntities.append(entity)
        entities.clear()
        for item in returnEntities:
            entities.append(item)






    def add_entities_if_synonyms(self, entity_a, entity_b):
        if entity_b is not None:
            original = utils.as_text_type(entity_a)
            replacement = utils.as_text_type(entity_b)

            #if original != replacement:
            # if (original in self.synonyms
            #         and replacement not in self.synonyms[original]):
            #         self.synonyms[original].append(replacement)
            if (original in self.synonyms
                    and self.synonyms[original] != replacement):
                warnings.warn("Found conflicting synonym definitions "
                              "for {}. Overwriting target {} with {}. "
                              "Check your training data and remove "
                              "conflicting synonym definitions to "
                              "prevent this from happening."
                              "".format(repr(original),
                                        repr(self.synonyms[original]),
                                        repr(replacement)))

            self.synonyms[original] = replacement
