from lib.ProcessText import *

text = []
#with open("ConvenioCabildo.pdf", encoding="utf8") as input:
text = ProcessText.readFile("ConvenioCabildo.pdf", "pdf")

#print(text)
with open("salida.txt", "w",encoding="utf-8") as text_file:
    text_file.write(text)
