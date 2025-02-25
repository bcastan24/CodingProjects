#A program to turn a pdf into an audiobook
#need to install pyttsx3 python package
#also need to install pypdf2 python package
import pyttsx3
import PyPDF2
book = open('oopbook.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(book)
pages = pdfReader.numPages
speaker = pyttsx3.init()
page = pdfReader.getPage(7)
text = page.extractText()
speaker.say(text)
speaker.runAndWait()