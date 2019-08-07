from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_nlu_gao.training_data import Message, TrainingData
from rasa_nlu_gao.training_data.formats.readerwriter import JsonTrainingDataReader

logger = logging.getLogger(__name__)


class LuisReader(JsonTrainingDataReader):

    def read_from_json(self, js, **kwargs):
        # type: (Text, Any) -> TrainingData
        """Loads training data stored in the LUIS.ai data format."""

        training_examples = []
        regex_features = []

        # Simple check to ensure we support this luis data schema version
        if not js["luis_schema_version"].startswith("2"):
            raise Exception("Invalid luis data schema version {}, should be 2.x.x. "
                            "Make sure to use the latest luis version "
                            "(e.g. by downloading your data again)."
                            "".format(js["luis_schema_version"]))

        synonyms_dict = {}

        lookup_tables = []

        close_list = js.get("closedLists",[])
        for close_item in close_list:
            name = close_item.get("name")
            subLists = close_item.get("subLists")
            for item in subLists:
                lookup_dict = {}
                elements_list = []
                if len(item.get("list")) > 0:
                    for synonyms_item in item.get("list"):
                        synonyms_dict[synonyms_item] = item.get("canonicalForm")
                        synonyms_dict[item.get("canonicalForm")] = item.get("canonicalForm")

                    lookup_dict["name"]=name
                    elements_list.append(item.get("canonicalForm"))
                    elements_list.extend(item.get("list"))
                    lookup_dict["elements"] = list(set(elements_list))
                    lookup_dict["resolution"] = item.get("canonicalForm")
                    lookup_tables.append(lookup_dict)

                else:
                    lookup_dict = {}
                    elements_list = []
                    lookup_dict["name"] = name
                    synonyms_dict[item.get("canonicalForm")] = item.get("canonicalForm")
                    elements_list.append(item.get("canonicalForm"))
                    lookup_dict["elements"] = list(set(elements_list))
                    lookup_dict["resolution"] = item.get("canonicalForm")
                    lookup_tables.append(lookup_dict)


        for r in js.get("regex_features", []):
            if r.get("activated", False):
                regex_features.append({"name": r.get("name"),
                                       "pattern": r.get("pattern")})

        for s in js["utterances"]:
            text = s.get("text")
            intent = s.get("intent")
            entities = []
            for e in s.get("entities") or []:
                start, end = e["startPos"], e["endPos"] + 1
                val = text[start:end]
                entities.append({"entity": e["entity"],
                                 "value": val,
                                 "start": start,
                                 "end": end})

            data = {"entities": entities}
            if intent:
                data["intent"] = intent
            training_examples.append(Message(text, data))
        return TrainingData(training_examples,entity_synonyms=synonyms_dict,regex_features=regex_features,lookup_tables=lookup_tables)
