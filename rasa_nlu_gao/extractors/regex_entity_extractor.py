from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import time

import requests
import simplejson
import re

from typing import Any
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu_gao.config import RasaNLUModelConfig
from rasa_nlu_gao.extractors import EntityExtractor
from rasa_nlu_gao.extractors.duckling_extractor import (
    filter_irrelevant_matches, convert_duckling_format_to_rasa)
from rasa_nlu_gao.model import Metadata
from rasa_nlu_gao.training_data import Message


logger = logging.getLogger(__name__)

#主要负责提取身份证
class RegexEntityExtractor(EntityExtractor):

    name="ner_regex_entity"
    provides = ["entities"]
    defaults={
        #正则表达式-数组类型---（映射到metadata.json时候 \d变成\\d)
        #用于多个表达式 提取实体 暂不实现
        #"regexs":[r"([1-9]\d{5}(18|19|([23]\d))\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]$)|(^[1-9]\d{5}\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\d{2}[0-9Xx])"],
        #启用False||True
        "startusing":False
    }

    def __init__(self, component_config=None, language=None):
        super(RegexEntityExtractor, self).__init__(component_config)
        self.language = language

    def _match(self,message):
        matchs=[]
        try:
            #regex=r"([1-9]\d{5}(18|19|([23]\d))\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]$)|(^[1-9]\d{5}\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\d{2}[0-9Xx])"
            regex=r"\d{6}(18|19|20)?\d{2}(0[1-9]|1[012])(0[1-9]|[12]\d|3[01])\d{3}(\d|[xX])"
            pattern = re.compile(r"{0}".format(regex), re.I)
            m = pattern.search(message.text)
            if m is not None:
                matchs.append((m.group(0), m.span(1)))
        except Exception as ex:
            logger.exception("error in parse function :" + ex)
            return matchs
        return matchs

    def convert_regex_format_to_rasa(self,matchs):
        extracted = []
        for match in matchs:
            entity = {"start": match[1][0],
                  "end": match[1][1],
                  "text": match[0],
                  "confidence": 1.0,
                  "value":  match[0],
                  "additional_info": match[0],
                  "entity":"idcard"}
            extracted.append(entity)
        return extracted

    def process(self, message, **kwargs):
        startusing=self.component_config["startusing"]
        if startusing:
            matchs=self._match(message)
            extracted=self.convert_regex_format_to_rasa(matchs)
        else:
            extracted = []
        extracted = self.add_extractor_name(extracted)
        message.set("entities",
                    message.get("entities", []) + extracted,
                    add_to_output=True)

    @classmethod
    def create(cls, config):
        # type: (RasaNLUModelConfig) -> 
        return RegexEntityExtractor(config.for_component(cls.name,
                                                          cls.defaults),
                                     config.language)

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  
             **kwargs  # type: **Any
             ):

        component_config = model_metadata.for_component(cls.name)
        return cls(component_config, model_metadata.get("language"))

