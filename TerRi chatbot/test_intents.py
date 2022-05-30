from intents import get_intent
from snips_nlu import SnipsNLUEngine
import os
from debug import log, out

list_of_queries = [
    "Is it going to rain today?",
    "How are you?",
    "What's the weather like today?",
    "I'm hungry!",
    "Who are you?",
    "Nearby tourist places",
    "What are some good restaurants nearby?",
    "Tell me a joke.",
    "Are there some good hotels nearby?",
    "Is there a good place for dining?",
    "Tell me some good places to see"
]

list_of_intents = [
    "weather",
    "casual",
    "weather",
    "restaurant",
    "casual",
    "places",
    "restaurant",
    "casual",
    "restaurant",
    "restaurant",
    "places",

]

if __name__ == "__main__":

    log('Intents Testing Program started.')

    count = 0
    score = 0
    
    engine = SnipsNLUEngine.from_path(os.getcwd()+'/model')

    for i in range(len(list_of_queries)):
        count += 1 
        query = list_of_queries[i]
        log('Query: ' + query)
        intent = get_intent(query, engine)
        log('Intent actual: ' + list_of_intents[i])
        if intent == list_of_intents[i]:
            score += 1
            pass
        pass

    accuracy = score / count

    out('Accuracy is ' + str(accuracy) + ' out of ' + str(len(list_of_intents)) + ' queries.')

    log('Intents Testing Program ended.')
    pass