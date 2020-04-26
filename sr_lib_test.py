#!/usr/bin/env python3

# Test for SpeechRecognition python library
# dependencies:
# - PyAudio==0.2.11
# - SpeechRecognition==3.8.1

import speech_recognition as sr

r = sr.Recognizer() # interface

with sr.Microphone() as source:
    print('Say something and wait for the transcription!')
    audio = r.listen(source) # capture mic input

# testing Google speech recognition API
# default API key is used
try:
    print(r.recognize_google(audio))
except sr.UnknownValueError:
    print('Google Speech Recognition API did not understand the input given')
except sr.RequestError as e:
    print('Error while requesting results from Google Speech Recognition service. Error msg: {}'.format(e))


