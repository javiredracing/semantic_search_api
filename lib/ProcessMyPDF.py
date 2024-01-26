# To read the PDF
import pypdf
# To analyze the PDF layout and extract text
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
# To extract text from tables in PDF
import pdfplumber
# To extract the images from the PDFs
from PIL import Image
from pdf2image import convert_from_bytes
# To perform OCR to extract text from images 
import pytesseract 
# To remove the additional created files
import os, io
#from datetime import datetime
#import time
#from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait

# Create function to extract text
class ProcessMyPDF:
    @staticmethod
    def text_extraction(element, line):
        # Extracting the text from the in line text element
        line_text = element.get_text()
        
        # Find the formats of the text
        # Initialize the list with all the formats appeared in the line of text
        line_formats = []
        for text_line in element:
            if isinstance(text_line, LTTextContainer):
                # Iterating through each character in the line of text
                for character in text_line:
                    if isinstance(character, LTChar):
                        # Append the font name of the character
                        #line_formats.append(character.fontname)
                        # Append the font size of the character
                        #line_formats.append(character.size)
                        line_formats.append({"fontname":character.fontname, "size":int(character.size), "line":line})
                        break   #only first character evaluated
                line = line + 1
        # Find the unique font sizes and names in the line
        #format_per_line = list(set(line_formats))
        
        # Return a tuple with the text in each line along with its format
        return (line_text, line_formats, line)

    # Convert table into appropriate fromat
    @staticmethod
    def table_converter(table):
        table_string = ''
        # Iterate through each row of the table
        for row_num in range(len(table)):
            row = table[row_num]
            # Remove the line breaker from the wrapted texts
            cleaned_row = [item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item in row]
            # Convert the table into a string 
            table_string+=('|'+'|'.join(cleaned_row)+'|'+'\n')
        # Removing the last line break
        table_string = table_string[:-1]
        return table_string

    # Create a function to check if the element is in any tables present in the page
    @staticmethod
    def is_element_inside_any_table(element, page ,tables):
        x0, y0up, x1, y1up = element.bbox
        # Change the cordinates because the pdfminer counts from the botton to top of the page
        y0 = page.bbox[3] - y1up
        y1 = page.bbox[3] - y0up
        for table in tables:
            tx0, ty0, tx1, ty1 = table.bbox
            if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
                return True
        return False

    # Function to find the table for a given element
    @staticmethod
    def find_table_for_element(element, page ,tables):
        x0, y0up, x1, y1up = element.bbox
        # Change the cordinates because the pdfminer counts from the botton to top of the page
        y0 = page.bbox[3] - y1up
        y1 = page.bbox[3] - y0up
        for i, table in enumerate(tables):
            tx0, ty0, tx1, ty1 = table.bbox
            if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
                return i  # Return the index of the table
        return None  
        
    # Create a function to crop the image elements from PDFs
    @staticmethod
    def crop_image(element, pageObj):
        # Get the coordinates to crop the image from PDF
        [image_left, image_top, image_right, image_bottom] = [element.x0,element.y0,element.x1,element.y1] 
        # Crop the page using coordinates (left, bottom, right, top)
        pageObj.mediabox.lower_left = (image_left, image_bottom)
        pageObj.mediabox.upper_right = (image_right, image_top)
        # Save the cropped page to a new PDF
        cropped_pdf_writer = pypdf.PdfWriter()
        cropped_pdf_writer.add_page(pageObj)
        buffer = io.BytesIO()
        cropped_pdf_writer.write(buffer)
        images = convert_from_bytes(buffer.getvalue(),fmt='png',poppler_path=r"C:\Users\jfernandez\Downloads\Release-23.11.0-0\poppler-23.11.0\Library\bin")
        buffer.close()
        return images[0]

    # Create a function to convert the PDF to images
    # def convert_to_images(input_file,):
        # images = convert_from_path(input_file, poppler_path=r"C:\Users\jfernandez\Downloads\Release-23.11.0-0\poppler-23.11.0\Library\bin")
        # image = images[0]
        # output_file = 'PDF_image.png'
        # image.save(output_file, 'PNG')
    @staticmethod
    def save_image(image):
        output_file = 'PDF_image.png'
        image.save(output_file, 'PNG')

    # Create a function to read text from images
    # def imageFile_to_text(image_path):
        # # Read the image
        # img = Image.open(image_path)
        # # Extract the text from the image
        # text = pytesseract.image_to_string(img)
        # return text
    @staticmethod   
    def image_to_text(img):    
        text = pytesseract.image_to_string(img)
        return text

    #------------------------------------------------------
    @staticmethod
    def processPage(pagenum, page):
        # Initialize the variables needed for the text extraction from the page
        global pdfReaded
        global pdf     
        
        page_text = []
        line_format = []
        text_from_images = []
        text_from_tables = []
        page_content = []
        # Initialize the number of the examined tables
        table_in_page= -1
        # Find the examined page
        page_tables = pdf.pages[pagenum]
        # Find the number of tables in the page
        tables = page_tables.find_tables()
        if len(tables)!=0:
            table_in_page = 0

        for table in tables:
            table_string = ProcessMyPDF.table_converter(table.extract())
            text_from_tables.append(table_string)

        # Find all the elements
        page_elements = [(element.y1, element) for element in page._objs]
        # Sort all the element as they appear in the page 
        page_elements.sort(key=lambda a: a[0], reverse=True)
        cont=0
        # Find the elements that composed a page
        for i,component in enumerate(page_elements):
            # Extract the element of the page layout
            element = component[1]
            table_found = ProcessMyPDF.find_table_for_element(element,page ,tables)
            # Check the elements for tables
            if table_in_page == -1:
                pass
            else:
                #if ProcessMyPDF.is_element_inside_any_table(element, page ,tables):
                if table_found is not None:
                    #table_found = ProcessMyPDF.find_table_for_element(element,page ,tables)
                    if table_found == table_in_page:# and table_found != None:    
                        page_content.append("\n"+text_from_tables[table_in_page]+"\n")
                        page_text.append('\ntable\n')
                        #line_format.append('table')
                        table_in_page+=1
                    # Pass this iteration because the content of this element was extracted from the tables
                    continue

            if table_found is None:
                # Check if the element is text element
                if isinstance(element, LTTextContainer):
                    # Use the function to extract the text and format for each text element
                    (line_text, format_per_line, line) = ProcessMyPDF.text_extraction(element,cont)
                    cont = line
                    # Append the text of each line to the page text
                    page_text.append(line_text)
                    # Append the format for each line containing text
                    line_format = line_format + format_per_line
                    page_content.append(line_text)


                #Check the elements for images
                if isinstance(element, LTFigure):
                    # Crop the image from PDF
                    pageObj = pdfReaded.pages[pagenum]   
                    image = ProcessMyPDF.crop_image(element, pageObj)
                    image_text = ProcessMyPDF.image_to_text(image)
                    text_from_images.append(image_text)
                    page_content.append(image_text)
                    # Add a placeholder in the text and format lists
                    page_text.append('\nFigure\n')
                    #line_format.append('image')
                    # Update the flag for image detection                    

        return {"page":pagenum, "pageText":page_text, "lineFormat":line_format, "textImages":text_from_images,"textTables":text_from_tables,"page_content":page_content}

    #---------------------------------------------------------------------------
    # initialize worker processes
    @staticmethod
    def init_worker(_pdfReaded, _pdf):
        # declare scope of a new global variable
        global pdfReaded
        global pdf
        # store argument in the global variable for this process    
        # Create a pdf reader object
        pdfReaded = _pdfReaded
        pdf = _pdf
    
    @staticmethod
    def readPDF(bytesFile):
    #if __name__ == '__main__':    
        #start = datetime.now()
        # Find the PDF path
        bytes_stream = io.BytesIO(bytesFile)
        #pdf_path = 'Metropolitano.pdf'
        pdf = pdfplumber.open(bytes_stream)
        pdfReaded = pypdf.PdfReader(bytes_stream)
        # Create the dictionary to extract text from each image
        text_per_page = {}
        # Create a boolean variable for image detection
        
        with ProcessPoolExecutor(initializer=ProcessMyPDF.init_worker, initargs=(pdfReaded,pdf,)) as executor:
            # submit tasks and collect futures
            #print(executor._max_workers)
            pages = extract_pages(bytes_stream)
            futures = [executor.submit(ProcessMyPDF.processPage, pagenum, page) for pagenum, page in enumerate(pages)]
            # process task results as they are available
            wait(futures)
            # process task results as they are available
            for future in futures:
                result = future.result()
                dctkey = 'Page_'+str(result["page"])
                # Add the list of list as value of the page key
                text_per_page[dctkey]= result


        bytes_stream.close()

        result = ""
        for i, item in enumerate(text_per_page.items()):
            result = result + ''.join(text_per_page['Page_' + str(i)]["page_content"])
            result = result + "\f\r\n"
                
        return result