#A program to turn a pdf into an audiobook
#need to install pyttsx3 python package
#also need to install pypdf2 python package
import pyttsx3
import PyPDF2
from PyPDF2 import PdfReader
book = open('oopbook.pdf', 'rb')
pdfReader = PdfReader(book)
pages = len(pdfReader.pages)
speaker = pyttsx3.init()
userPage = input("Enter what page you would like to read: ")
pageToRead = int(userPage)
if pageToRead >= 0 and pageToRead < pages:
    page = pdfReader.pages[pageToRead]
    text = page.extract_text()
    speaker.say(text)
    speaker.runAndWait()