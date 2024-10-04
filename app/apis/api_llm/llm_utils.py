TEMPLATE_ASK="""
Se proporciona información procedente de cada documento:
    
{% for name, content in myDocs.items() %}
    Documento: {{ name }}        
    {% for text in content %}
        - {{text}}
        
    {% endfor %}   
{% endfor %}
    
Por cada uno de los documentos proporcionados, responde a la siguiente pregunta en el mismo idioma: {{ query }}. No te inventes nada.
    """

TEMPLATE_SUMMARY = """
Haz un resumen conciso y claro que capture los puntos más importantes del siguiente contexto. No te inventes nada. Responde solo con el resumen. Si no puedes hacerlo, no pongas nada.
Contexto:
{{myDocs}}

Resumen:
"""

TEMPLATE_TRANSLATE = """
La siguiente lista contiene textos:

{{srt_file}}

Responde con una lista con cada texto traducido al idioma {{lang}}, en el mismo orden y con igual formato que la lista original. No te inventes nada.

Ejemplo respuesta:

1- Hello world.

2- A long sentence to translate.

"""