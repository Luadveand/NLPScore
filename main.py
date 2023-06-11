import nltk
import unicodedata
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from nltk.tokenize import word_tokenize


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


# Descarga de los datos necesarios para la lematización
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 2))


def calcular_puntaje(perfil_profesional, oferta_laboral):
    habilidades = [
        ".net", "administración de bases de datos", "agile", "ajax", "analista", "android", "angular", "angular react",
        "angularjs", "apache kafka", "api design", "apollo", "arquitectura mvc", "asp .net mvc 5", "asp.net mvc", "aws",
        "azure", "azure devops", "babel", "base de datos sql db2", "blackduck", "blockchain", "bootstrap", "c",
        "c#", "c++", "circleci", "confluence", "conocimientos en redes", "css", "css", "d", "deploy de aplicaciones",
        "desarrollo de apis", "desarrollo de servicios", "devops", "docker", "documentación", "elasticsearch",
        "elaboración de informes",
        "entrega continua", "erp gp dynamics", "ethereum", "express.js", "firebase", "figma", "flask", "fortify",
        "framework 4.5", "gcp", "git", "go", "graphql", "hadoop", "hibernate", "html", "html", "inglés intermedio",
        "integración continua", "ios", "java", "javascript", "javascript", "jenkins", "jira", "jquery", "jquery", "jsp",
        "json", "kanban", "keras", "kotlin", "kubernetes", "laravel", "less", "linux", "mantenimiento de sistemas",
        "metodologías ágiles", "microservices", "microsoft sql server", "modelamiento de bases de datos", "mongodb",
        "mysql",
        "nginx", "node.js", "nodejs", "nosql", "oauth", "openapi", "oracle db", "owasp", "pentesting", "php",
        "postgresql", "postman", "powershell", "producir código", "programador", "pytorch", "python", "rabbitmq",
        "rails",
        "react", "redis", "redux", "registro de actividades", "remoto/híbrido", "responsive design", "rest", "ruby",
        "rust", "sass", "scala", "scrum", "security", "selenium", "servlet", "shell scripting", "smart contracts",
        "soap", "soap ui", "solidity", "sonar q", "spark", "spring", "sqlite", "sql server", "sts", "subversion",
        "support", "swagger", "symfony", "t", "tensorflow", "testing", "trello", "travis ci", "transact sql",
        "typescript", "ui/ux", "unix", "visual studio", "vue.js", "web accessibility", "web api", "web inspect",
        "web services",
        "webforms", "webforms", "webpack", "websockets", "winforms", "análisis de requerimientos", "soporte técnico",
        "servicios rest"
    ]

    # Preprocesamiento: remover saltos de línea, guiones, tildes y caracteres adicionales
    perfil_profesional = remove_accents(perfil_profesional.replace("\n", " ").replace("-", " "))
    oferta_laboral = remove_accents(oferta_laboral.replace("\n", " ").replace("-", " "))

    # Remover caracteres adicionales
    additional_chars = '[\\/"()@]'
    perfil_profesional = re.sub(additional_chars, '', perfil_profesional)
    oferta_laboral = re.sub(additional_chars, '', oferta_laboral)

    # Imprimir textos antes de lematizar
    print("Texto de perfil profesional antes de la lematización:")
    print(perfil_profesional)
    print("\nTexto de oferta laboral antes de la lematización:")
    print(oferta_laboral)

    # Convertir a minúsculas y lematizar
    perfil_profesional = " ".join([lemmatizer.lemmatize(word.lower()) for word in perfil_profesional.split()])
    oferta_laboral = " ".join([lemmatizer.lemmatize(word.lower()) for word in oferta_laboral.split()])

    # Transformar texto a vectores, considerando unigramas y bigramas
    perfil_vector = vectorizer.fit_transform([perfil_profesional])
    oferta_vector = vectorizer.transform([oferta_laboral])

    # Calcular similitud de coseno
    similitud = cosine_similarity(oferta_vector, perfil_vector)[0][0] * 100

    # Calcular frecuencias de habilidades
    frecuencia = Counter([word for word in word_tokenize(perfil_profesional) if word in habilidades])

    # Calcular puntuación de habilidades
    puntuacion_habilidades = [(h, (frecuencia[h] / len(word_tokenize(perfil_profesional))) * 1000) for h in
                              frecuencia.keys()]

    suma_puntuaciones = 0
    for habilidad, puntuacion in puntuacion_habilidades:
        print(f"La habilidad '{habilidad}' tiene una puntuación de: {puntuacion}")
        suma_puntuaciones += puntuacion

    print(f"La suma total de las puntuaciones de habilidades es: {suma_puntuaciones}")

    # Combinar similitud y puntuación de habilidades
    puntaje_total = similitud * 0.8 + suma_puntuaciones * 1.2

    print(f"Puntaje de similitud de coseno: {similitud * 0.8:.2f}")
    print(f"Suma de puntuaciones de habilidades: {suma_puntuaciones * 1.2:.2f}")
    print(f"Puntaje total (suma de ambos): {puntaje_total:.2f}")

    return puntaje_total


# Ejemplo de uso
perfil_profesional = """
Zt iy\n\nLuis Adrian\nVelasco Andaluz\n\nEstudiante Ingenieria de Software\n\n9 Lima, Pert\ne 17/01/1997\nIvelascoand@gmail.com\n\n\\ 964665279\n\nin Luis Adrian Velasco Andaluz\n\n© Luadveand\n\nEspanol Inglés\n\nNativo Avanzado\nHABILIDADES\n\nHTML, CSS\nBe\nJavascript\nLs\nC#/.Net\nCe\nSQL, NoSQL\nee\n\nHabilidades blandas:\nTrabajo en equipo\nCapacidad analitica\nResolucion de problemas\n\nBuena actitud\n\nINTERESES\nMusica Patinaje urbano\nLectura Debatir\n\nHerramientas\nproductividad\n\nTecnologias\nemergentes\n\nSoy un programador Full-Stack Junior con experiencia en desarrollo de\nsoluciones aplicando metodologias agiles y conocimiento en distintas\ntecnologias. Me caracterizo por mi personalidad sociable, gran capacidad\nde comunicacién y entendimiento sobre distintos temas. Me encantaria\ntrabajar con ustedes y poder demostrar todas mis capacidades.\n\nEDUCACION\n\nColegios Pamer (16/03/2011 - Instituto Britanico (16/03/2011 -\n\nSecundaria 20/12/2013) Intermedio - 13/12/2012)\n\nCompleta Avanzado, Inglés\n\nUniversidad Peruana de (15/03/2018- Asociacién Pandemia — (99/01/2019 -\nActualidad) 28/02/2019)\n\nCiencias Aplicadas\nPregrado, Ingenieria de\nSoftware\n\nCurso, Community\nManager y Publicidad\nDigital\n\nEXPERIENCIA LABORAL\n\nPizza de la Plaza (02/05/2022 - Actualidad)\n\nDesarrollador Full-Stack\n\nMe dediqué al desarrollo de la idea, disefio y parte de la\nprogramacién de una PWA para un restaurante pizzeria que solucione\nel manejo de informaci6n de todas las érdenes. Actualmente me\nconsidero en el constante mantenimiento del mismo.\n\nEl proyecto fue desarrollado con React para el lado del frontend,\nFirebase Realtime Database para el backend y el servicio de hosting\nincluido en Firebase.\n\nGrupo UROTOYA (05/07/2022 - Actualidad)\n\nSoporte TI\n\nMi rol es el de asistir ante cualquier incdgnita respecto a problemas o\nsoluciones tecnoldgicas que puedan aparecer en la empresa, como\ntambién asesorar sobre el correcto manejo de distintas herramientas\nweb y posibles tecnologias que pudieran usar.\n\nMe encargo también del manejo de los dominios como correos
electrónicos y mantenimiento de los sitios web.\n\nPROYECTOS\n\nMecanillama (16/05/2022 - 16/10/2022)\n\n-NET, MySQL\n\nProyecto universitario de solucién de contacto y reserva de citas entre\ntalleres mecanicos y clientes que necesitan un servicio automotriz.\nImplementa el completo funcionamiento de un backend desarrollado\nen .NET usando Entity Framework, una base de datos MySQL,\nautenticaciédn con JWT y proceso DevOps.\n\n@ https://github.com/Mecanillama-WX71/Mecanillama-backend-v2\n\nLawyeed (16/09/2022 - 16/11/2022)\n\n-NET, MySQL\nProyecto universitario de solucién de contacto entre abogados y\nclientes que necesiten asesoria legal. Implementa el completo\nfuncionamiento de un backend desarrollado en .NET usando Entity\nFramework y base de datos MySQL.\n\n© https://github.com/PixiUPC/BackEnd\n
"""

oferta_laboral = """
"PROGRAMADOR JR.(Web, Asp.net, C#, sql Microsoft) GUARDAR S Selectiva Lima Postularse directamente en Indeed Tiempo completo Nuestro cliente, importante empresa que brinda software de planillas que facilitan la gestión del personal, nos ha solicitado el servicio de HEADHUNTING y ubicar a un profesional de primer nivel para cubrir la siguiente posición: NOMBRE DE LA POSICIÓN: ANALISTA JUNIOR PROGRAMADOR WEB DOMINIO(S) PARTICULAR(ES) DEL CANDIDATO: DOMINIO DE MICROSOFT.NET Y PROGRAMACIÓN WEB CON VISUAL STUDIO.NET Y MICROSOFT SQL SERVER GIRO DE EMPRESA: COMERCIALIZADORA DE SOFTWARE DE PLANILLAS Y GESTIÓN DE PERSONAL ZONA 1. COMPETENCIAS DURAS 1.1 ACADÉMICOS: Profesional egresado de: Ingeniería de sistemas, técnico en computación y sistemas Titulado o similares. 1.2 GIROS DE PROCEDENCIA: Provenir muy deseablemente de empresas con giros de negocio dedicadas al desarrollo de software financiero, contable y/o planillas; sin embargo otros giros serán considerados, siempre y cuando el candidato cuenta con la experiencia solicitada para esta posición. 1.3 AREAS DE PROCEDENCIA: Provenir, deseablemente, del área de: Tecnologías de la información y/o Sistemas. 1.4 TIEMPO MINIMO DE EXPERIENCIA: Demostrar, obligatoriamente, tener no menos de 1 año ocupando una posición igual o similar a la solicitada desarrollada en el rubro(s): 1.5 OTROS DOMINIOS ESPERADOS: Obligatorios: C# MICROSOFT SQL SERVER (STORE PROCEDURES NIVEL AVANZADO) Net Framework 4 o superior (deseable.Net Core) ASP.NET MVC WebForms / WinForms Experiencia en Frontend & Backend. Bootstrap / CSS html5/js/JQuery Deseable: Desarrollo para móviles Conocimientos en git- tfs Contar con portafolio de proyectos realizados bajo su cargo. ZONA 2. FUNCIONES Y RESPONSABILIDADES 2.1 DE REPORTE (A quien reporta y que tipo de reportes): · Reportar al Jefe de Soporte y Operaciones los informes de avance semanal de acuerdo al cronograma establecido. · Reportar a gerentes y socios los cumplimientos de objetivos y desarrollo de actividades. 2.2 DE COORDINACION (Con quien Coordina y que coordina): · Coordinar con el Jefe de soporte, desarrolladores y Gerentes la presentaciones de los entregables, QA · Coordinar con sus pares el intercambio de información y temas de intereses internos de la empresa. 2.3 DE GESTION (Con quien Gestiona y que gestiona Clientes, Proveedores, Gobierno): · Realizar pruebas de calidad y funcionalidad de la página web con algunos clientes y gestión con el equipo interno de trabajo. · Culminar el proyecto de la nueva versión del Software de planillas de la empresa. · Manejar proyectos de visión web. · Elaborar y presentar el cronograma de avance. ZONA 3. COMPETENCIAS BLANDAS · Liderazgo · Innovación · Orientación al cliente · Orientación a resultados · Visión de negocio · Orden y organización · Orientación a la calidad · Iniciativa y proactividad · Trabajo en equipo · Capacidad de negociación y persuasión · Resiliencia",
"""

print(calcular_puntaje(perfil_profesional, oferta_laboral))
