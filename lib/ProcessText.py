import fitz
import os, io
from thefuzz import fuzz

#from lib.ProcessMyPDF import *

class ProcessText:

    VALID_FORMATS_FITZ = ["pdf","xps", "epub", "mobi", "fb2", "cbz", "svg"]
    VALID_FORMATS = ["txt"]
    
    @staticmethod
    def check_if_header(pages):
        print("pages", len(pages))
        if len(pages) > 1:    
            pages[0] = pages[0].strip()
            first_page = pages[0].splitlines()
            print(first_page[0])
            for idx, line in enumerate(first_page): 
                for s in pages[1:]:
                     s = s.strip()
                     next_page = s.splitlines()
                     #print("line: "+str(idx) + "|* page0*|: " + line + " |*page"+ "*|: "+ next_page[idx])
                     if len(next_page) <= idx or fuzz.ratio(next_page[idx],line) < 90:
                        return idx
        return 0
        
    @staticmethod    
    def check_if_footer(pages):
        if len(pages) > 1:    
            pages[0] = pages[0].strip()
            first_page = pages[0].splitlines()
            idx = 0
            for line in reversed(first_page):
                for s in pages[1:]:
                    s = s.strip()
                    next_page = s.splitlines()                
                    current_line = len(next_page) - idx - 1 
                    if current_line < 0 or fuzz.ratio(next_page[current_line],line) < 90:
                        return idx
                idx+=1
        return 0
        
    @staticmethod
    def remove_header_footer(text):
        '''Remove header and footer'''
        pages = text.split("\f")
        #pages = [x for x in pages if len(x.strip())>0] #remove white slices
        header = ProcessText.check_if_header(pages)
        footer = ProcessText.check_if_footer(pages)
        if header > 0 or footer > 0:
            text_formatted = ""
            for page in pages:            
                #text_formatted = text_formatted + remove_header_footer(page, header, footer)
                page = page.strip()
                text_splitted = page.splitlines()
                result = ""
                value = len(text_splitted)
                if footer > 0:
                    value = -1 * footer
                for line in text_splitted[header:value]:
                    text_formatted = text_formatted + line + "\r\n"                
                text_formatted = text_formatted + "\f\r\n"
            
            return text_formatted.strip()
        else:
            return text

    @staticmethod
    def readFile(bytesFile, fileType):
        plain_text = ""                  
        fileType = fileType.lower() 
        #bytesFile = file.file.read()            
        if fileType in ProcessText.VALID_FORMATS_FITZ:
            plain_text = ProcessText.readFileFitz(bytesFile, fileType)
        elif fileType in ProcessText.VALID_FORMATS:
            plain_text = bytesFile.decode("utf-8")                
        #elif fileType in ProcessText.PDF_FORMAT:
        #    plain_text = ProcessMyPDF.readPDF(bytesFile)
        else:
            print("Error extension")
            return plain_text
        txt_formatted = ProcessText.remove_header_footer(plain_text)
        return txt_formatted
                    
    @staticmethod        
    def readFileFitz(bytesFile, myformat):
        #TODO: https://towardsdatascience.com/extracting-text-from-pdf-files-with-python-a-comprehensive-guide-9fc4003d517
        doc = fitz.open(stream=bytesFile, filetype=myformat)
        plain_text = ""
        
        for page in doc: 
        
            # for tab in page.find_tables():
                # # process the content of table 'tab'
                # page.add_redact_annot(tab.bbox)  # wrap table in a redaction annotation
            # page.apply_redactions()  # erase all table text
        
            blocks = page.get_text("blocks",sort=True)
            for block in blocks:
                if block[6] == 0: #0 text, 1 image
                    plain_text += block[4]  #4 = text
                    item = block[4].rstrip()
                    if len(item) > 0 and item[-1] in [".","?","!"]:
                        plain_text += "\r\n"
            plain_text = plain_text + "\f\r\n"
        return plain_text.strip()
    
    @staticmethod    
    def chunk_text(input):
        '''
        parse text by empty line. clear emtpy lines and whites in the end.
        '''
        text = ""
        splitted_text = []
        text_page = []
        buf = io.StringIO(input)
        file_content = buf.readlines()
        page_counter = 1
        for line in file_content:
            #line = line.decode("utf-8")       
            aux = line.strip()
            if len(aux) == 0 :  #breakline
                if len(text) > 0 and text.strip().endswith("."):
                    splitted_text.append(text.strip())                
                    text_page.append(page_counter)
                    text = ""
                if "\f" in line:    #end of page
                    page_counter+=1
                #continue
            else:
                if not aux.endswith(".") and not aux.endswith(":"):
                    line = aux + " "
                text = text + line

        return splitted_text, text_page