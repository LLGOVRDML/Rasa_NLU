FROM tensorflow/tensorflow:1.12.0-py3
COPY . /usr/local/src/
RUN pip3 install -r /usr/local/src/alt_requirements/requirements_bare.txt -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install /usr/local/src/zh_core_web_sm-2.0.5.tar.gz
RUN python3 -m spacy link zh_core_web_sm zh
RUN chmod +x /usr/local/src/entrypoint.sh
ENTRYPOINT ["/usr/local/src/entrypoint.sh"]

