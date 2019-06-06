# HT_NLU
High talk nalture language understanding platform of LingLing technology . Ltd


# LingLing Technology Dalian Team's project

## Sering bert-as-service model<br>

1、clone HT_NLU project to the local machine <br>
```
git clone https://github.com/LLGOVRDML/HT_NLU.git
```

2、Serving bert as service model as service<br>
 cmd -->  Windows+R
```
pip install bert-serving-server -i https://mirrors.aliyun.com/pypi/simple
pip install bert-serving-client -i https://mirrors.aliyun.com/pypi/simple
cd ${yourpath}/HT_NLU/bert-as-service
bert-serving-start -model_dir D:\chinese_L-12_H-768_A-12 -tuned_model_dir C:\Users\weizhen\Desktop\NLU\rasa_model_output -ckpt_name=model.ckpt-1028
```
 ![image](https://github.com/LLGOVRDML/HT_NLU/raw/master/bert_start.PNG)
when you see the log print "ready and listening" it means that the bert server is ready , and we can go to the next step <br>




## Start rasa_nlu_gq server for classification process <br>


0、cd the root project folder , and double click "visualcppbuildtools full.exe" to install the c++ compiler in windows<br>





1、install related python packages<br>
```
cd ${yourpath}/HT_NLU/rasa_nlu_gq
pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
pip install -U bert-serving-client -i https://mirrors.aliyun.com/pypi/simple
pip install spacy==2.0.16 -i https://mirrors.aliyun.com/pypi/simple
pip install /usr/local/src/zh_core_web_sm-2.0.5.tar.gz -i https://mirrors.aliyun.com/pypi/simple
python -m spacy link zh_core_web_sm zh
pip install -r ./requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install tensorflow -i https://mirrors.aliyun.com/pypi/simple
pip install jieba -i https://mirrors.aliyun.com/pypi/simple
pip install GPUtil -i https://mirrors.aliyun.com/pypi/simple
pip install sklearn_crfsuite==0.3.6 -i https://mirrors.aliyun.com/pypi/simple
pip install scikit-learn==0.19.2 -i https://mirrors.aliyun.com/pypi/simple
pip install numpy -i https://mirrors.aliyun.com/pypi/simple
pip install scipy -i https://mirrors.aliyun.com/pypi/simple
```

2、open project in pycharm and edit execution path<br>
after the previous step , you can open the rasa_nlu_gq project in the pycharm ide <br>
and edit the configuration<br>


in pycharm edit configuration 

serving paramaters :
```
-c sample_configs/config_embedding_bert_intent_estimator_classifier.yml --path projects/bert_gongan_v4
```
training parameters:
```
-c sample_configs/config_embedding_bert_intent_estimator_classifier.yml --data data/examples/luis/HighTalkSQSWLuisAppStaging-GA-20180824.json --path projects/bert_gongan_v4
```

 ![image](https://github.com/LLGOVRDML/HT_NLU/raw/master/edit_config.PNG)
 
3、start the rasa_nlu_gq server<br>
and press the run button for running<br>

 ![image](https://github.com/LLGOVRDML/HT_NLU/raw/master/starting_rasa.PNG)
 
 
## test via browser<br>
 ![image](https://github.com/LLGOVRDML/HT_NLU/raw/master/test_via_browser.PNG)


#### train rasa process<br>
train rasa nlu with the bert words vectors
```
python train.py -c sample_configs/config_embedding_bert_intent_classifier.yml --data data/examples/luis/HighTalkSQSWLuisAppStaging-GA-20180824.json --path projects/bert_gongan_v4
```


## Deployed in docker<br>
#### build rasa docker image<br>
```
cd rasa_nlu_gq
docker build -t rasa_nlu_gq:v1.0 .

```
#### build bert-as-service docker image<br>
```
cd bert-as-service
docker build -t bert-as-service:v1.0 .

```
#### first run bert-as-service image<br>
```
docker run -it -p 5555:5555 -p 5556:5556 bert-as-service:v1.0

```
#### changing rasa_nlu_gq model's ip endpoint and rasa project's ip call out endpoint , after that you can run the rasa docker image<br>
```
docker run -it -p 5000:5000 rasa_nlu_gq:v1.0

```
