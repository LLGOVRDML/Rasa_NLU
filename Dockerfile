# base image
FROM centos

# MAINTAINER
MAINTAINER weizhen_zhao@163.com

COPY . /usr/local/src/

#ENV YUMREPOS /etc/yum.repos.d/
#WORKDIR ${YUMREPOS}
#RUN yum -y install wget
#RUN wget http://mirrors.163.com/.help/CentOS6-Base-163.repo
#RUN mv CentOS6-Base-163.repo CentOS-Base.repo
#RUN yum -y update

ENV SRC /usr/local/src/

# install python3 preconditional package
WORKDIR ${SRC}


# unpack python
RUN tar -xzvf ./Python-3.5.6.tgz

# install python
ENV PYTHONDIR /usr/local/src/Python-3.5.6
WORKDIR ${PYTHONDIR}
RUN yum -y install gcc make openssl-devel bzip2-devel expat-devel gdbm-devel readline-devel sqlite-devel

RUN ./configure --prefix=/usr/local/python-3.5.6

RUN echo `pwd`
RUN make && make install
# set soft link for python3 and pip3

ENV root /
WORKDIR ${root} 
RUN ln -s /usr/local/python-3.5.6/bin/python3 /usr/bin/python3
RUN ln -s /usr/local/python-3.5.6/bin/pip3 /usr/bin/pip3

RUN pip3 install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install -U bert-serving-client -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install spacy==2.0.16 -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install /usr/local/src/zh_core_web_sm-2.0.5.tar.gz -i https://mirrors.aliyun.com/pypi/simple
RUN python3 -m spacy link zh_core_web_sm zh
RUN yum -y install unzip



WORKDIR ${SRC}
RUN pip3 install Twisted -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install -r ./requirements.txt -i https://mirrors.aliyun.com/pypi/simple


# starting bert server
WORKDIR ${root}
RUN pip3 install tensorflow -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install jieba -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install GPUtil -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install sklearn_crfsuite==0.3.6 -i https://mirrors.aliyun.com/pypi/simple

COPY ./entrypoint.sh /usr/local/src/
RUN chmod +x /usr/local/src/entrypoint.sh
RUN pip3 install scikit-learn==0.19.2 -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install numpy -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install scipy -i https://mirrors.aliyun.com/pypi/simple
RUN echo `pwd`
WORKDIR ${SRC}
RUN echo `ls`
ENTRYPOINT ["/usr/local/src/entrypoint.sh"]

