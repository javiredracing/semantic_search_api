TEMPLATE_ASK="""
La siguiente lista contiene textos:

{{srt_file}}

Responde con una lista con cada texto traducido al idioma {{lang}}, en el mismo orden y con igual formato que la lista original. No te inventes nada.

Ejemplo respuesta:

1- Hello world.

2- A long sentence to translate.

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