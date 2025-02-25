#A program to turn a pdf into an audiobook
#need to install pyttsx3 python package
import pyttsx3
speaker = pyttsx3.init()
speaker.say("Hello World")
speaker.runAndWait()