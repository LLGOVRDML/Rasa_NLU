FROM tensorflow/tensorflow:1.12.0-py3
#FROM rasa_nlu_ssoc:3.0
COPY . /usr/local/src/
RUN pip install -r /usr/local/src/requirements.txt -i https://mirrors.aliyun.com/pypi/simple
RUN pip install /usr/local/src/zh_core_web_sm-2.0.5.tar.gz -i https://mirrors.aliyun.com/pypi/simple
RUN python3 -m spacy link zh_core_web_sm zh --force
RUN chmod +x /usr/local/src/entrypoint.sh
ENTRYPOINT ["/usr/local/src/entrypoint.sh"]