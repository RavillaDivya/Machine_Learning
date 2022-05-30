from operator import mod
from pkg_resources import parse_requirements
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN
from snips_nlu.pipeline.configs import (ProbabilisticIntentParserConfig, NLUEngineConfig)
import os
import json
import speech_recognition as sr
import pyttsx3
import shutil
import audio
from debug import log

def train_model(trainFile, opFile):
    os.system("snips-nlu download en")
    os.system("snips-nlu generate-dataset en "+ trainFile +">" + opFile)
    parser_config = ProbabilisticIntentParserConfig()
    engine_config = NLUEngineConfig([parser_config])
    nlu_engine = SnipsNLUEngine(engine_config)
    training_data = json.load(open(opFile,"r"))
    #pprint.pprint(training_data)
    parse_requirements("engine trained on intent data")
    nlu_engine = nlu_engine.fit(training_data, force_retrain=True)
    model_path = os.getcwd()+'/model'
    if os.path.isdir(model_path):
        shutil.rmtree(model_path, ignore_errors=True)
    nlu_engine.persist(os.getcwd()+'/model')
    return nlu_engine

def get_intent(sentence, nlu_engine):  
    intent = nlu_engine.parse(sentence)
    # log(intent['intent']['intentName'])
    log('Intent probability: ' + str(intent['intent']['probability']) )
    if intent['intent']['intentName'] == None:
        log('Intent name: casual')
        return 'casual'
    log('Intent name: ' + str(intent['intent']['intentName']) )
    return intent['intent']['intentName']

if __name__ == '__main__':
    log("Intents Program started.")

    # ---TRAIN ENGINE (Only if intents.yaml is changed)----
    # engine = train_model('intents.yaml','dataset.json')

    # ----USE EXISTING ENGINE-----
    engine = SnipsNLUEngine.from_path(os.getcwd()+'/model')

    # query = 'Can you suggest some places to eat mexican food'
    query = audio.listen()
    
    intent = get_intent(query, engine)
    audio.speak(intent)
    log("Intents Program ended.")