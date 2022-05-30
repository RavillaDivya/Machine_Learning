#region Imports
from asyncio.windows_events import NULL
from distutils.log import debug
import json
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN
from snips_nlu.pipeline.configs import (ProbabilisticIntentParserConfig, NLUEngineConfig)
import os
import geocoder
import requests
import intents
import audio
from debug import log, err, out
import argparse
import debug
# import casual
#endregion

#region RestaurantImports
import reviews_analysis
#endregion

#region Keys
API_KEY = 'd7a295f0a13a4bd59c0f9c670a5589b3'
W_API_KEY = '7c3a018b87f5f0f543830f149f24252f'
#endregion

#region Functions

def read_out(restaurants):
    pass

def execute(intent, query):

    ''' Executing switchboard for intent '''

    # audio.speak('I have identfied the intent of this query as : ' + intent)

    # Casual Intent
    if intent == 'casual':
        log('Casual intent implementation in progress...')

        # response = casual.get_response(query)

        # audio.speak(response)

        pass

    else:

        # Fetching coordinates for the remaining intents
        log('Fetching coordinates.')
        coords = geocoder.ip('me')
        log('Coordinates data', coords)
        
        if intent == 'weather':
            URL = 'https://api.openweathermap.org/data/2.5/weather?lat=' + str(coords.lat) + '&lon=' + str(coords.lng) + '&appid=' + W_API_KEY
            response = requests.get(URL)
            json_data = json.loads(response.text)
            log('Weather api response', json_data)
            
            try:
                description = json_data['weather'][0]['description']
                temperature = json_data['main']['temp']
            except Exception as e:
                err('Error on processing weather data.', e)
                # Need to announce an error message here before returning.
                return

            temperature_farenheit = (temperature - 273.15) * (9/5) + 32
            temperature_farenheit = round(temperature_farenheit, 2)

            log('Weather description', description)
            log('Weather temperature in farenheit', temperature_farenheit)

            audio.speak('The weather prediction, in your area is ' + description + ', with a temperature of ' + str(temperature_farenheit) + ' degree farenheit.')
                
            pass

        elif(intent == 'restaurant'):
            # restuarants = []
            # URL = 'https://api.geoapify.com/v2/places?categories=catering.restaurant&filter=circle:' + str(coords.lng) \
            #     + ',' \
            #     + str(coords.lat) + ',10000' +\
            #     '&limit=10&apiKey=' + API_KEY
            # response = requests.get(URL)
            # json_data = json.loads(response.text)
            # # print(json_data)

            # for feature in json_data['features']:
            #     restuarants.append(feature['properties']['name'])
            #     read_out(restuarants)

            # log(restuarants, restuarants)

            # audio.speak('Please find a list of nearby restaurants below.')
            # out(restuarants, 'Terri')

            place_name,positive_review, list_of_restaurants = reviews_analysis.get_review(coords.lat, coords.lng)
            
            # audio.speak('Please find a list of nearby restaurants below.')
            # out(list_of_restaurants, 'Terri')

            log(place_name)
            positive_review = positive_review.replace('\n', '')
            log(positive_review)

            audio.speak('You can try this restaurant, ' + place_name)
            audio.speak('This is what one person had to say about the restaurant.')
            audio.speak(positive_review)

            pass

        elif intent == 'places':
            places = []
            URL = 'https://api.geoapify.com/v2/places?categories=tourism.attraction&filter=circle:' + str(coords.lng) \
                + ',' \
                + str(coords.lat) + ',10000' + \
                '&limit=10&apiKey=' + API_KEY
            response = requests.get(URL)
            json_data = json.loads(response.text)

            for feature in json_data['features']:
                places.append(feature['properties']['name'])
                read_out(places)

            log(places, places)

            audio.speak("Here's a list of interesting places nearby")

            list_of_places = ", ".join(places)
            out(list_of_places, "Terri")

        else:
            err('Execution not present for intent')
            pass
        pass
    pass

def control_loop():

    activated = True
    termination_commands = ["you're terminated", "you are terminated", "exit", "stop", "thank you", "thanks", "cancel", "cancel that", "alright"]

    # Loading intent model
    try:
        engine = SnipsNLUEngine.from_path(os.getcwd()+'/model')    
    except Exception as e:
        err('Error on trying to load intents model.')
        err(e)

    while activated:
        
        try: 
            # Getting text query from audio
            query = audio.listen()
            # audio.speak('What I heard is: ' + query)
            out(query, 'User')

            # Check for termination
            if query in termination_commands:
                activated = False # More of a gesture.
                break
            
            # Getting intent from text
            intent = intents.get_intent(query, engine).lower()
            # log('Intent identified as : ' + intent)

            # Executing command based on intent
            execute(intent, query)
        except Exception as e:
            err('Unexpected error has occured in the control loop.')
            err(e)
        pass

    return

def main():

    # Initializing
    intro = "Hello! I'm Terri, how can I help you today?"
    audio.speak(intro)

    control_loop()

    # Exiting
    outro = "Thank you for interacting with me, have a nice day!"
    audio.speak(outro)
    
    pass

#endregion

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--verbose')
        args = parser.parse_args()
        # print("args.verbose", args.verbose)
        if args.verbose is not None:
            verbose = bool(int(args.verbose))
            if verbose:
                debug.set_verbose()

        log('Main Program started.')
        main()
        log('Main Program ended.')
    except Exception as e:
        err('Unexpected error has occured in the main program.')
        err(e)
    pass
