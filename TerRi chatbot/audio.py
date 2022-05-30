
import speech_recognition as sr
import pyttsx3
from debug import log, err, out

def main():
    text = listen()
    speak(text)

def listen():
    
    # Initialize the recognizer
    r = sr.Recognizer()
        
    # Loop infinitely till no errors
    
    while(1):   

        log('Starting Loop.')
        
        try:
            
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level
                log('Adjusting for ambient noise.')
                r.adjust_for_ambient_noise(source2, duration=0.2)
                
                #listens for the user's input
                log('Listening for input...')
                audio2 = r.listen(source2)
                
                # Using google to recognize audio
                log('Recognizing audio...')
                text = r.recognize_google(audio2)
                text = text.lower()
    
                log("Recognized audio : " + text)

                return text
                
        except sr.RequestError as e:
            err("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            err("unknown error occured")

        except Exception as e:
            err("Unknown error outside scope.")
            print(e)

def speak(text):

    log('Speaking text...')
    try:
            
        # log('Initializing text-to-speech engine')
        # Initialize the engine
        engine = pyttsx3.init()

        # log('Starting say function.')
        engine.say(text)
        out(text, 'Terri')

        # log('Starting runAndWait() function')
        engine.runAndWait()

    except Exception as e:
        err("Unknown error outside scope.")
        print(e)
    pass

if __name__ == '__main__':
    log("Audio Program started.")
    main()
    log("Audio Program ended.")