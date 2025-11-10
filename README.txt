================================================================================
   APP RAG - VIDEOS Y DOCUMENTOS
   Sistema de Consulta Inteligente con IA
================================================================================

DESCRIPCIÓN
-----------
Aplicación web que permite cargar y consultar documentos (PDF, DOCX, TXT) y 
videos (MP4) mediante un asistente de IA. Utiliza Azure OpenAI para procesar
las consultas y generar respuestas contextuales basadas en el contenido.

Características principales:
- Carga de archivos PDF, DOCX, TXT y MP4
- Procesamiento de video con extracción de frames y OCR
- Chat interactivo con IA basado en el contenido
- Visualización de páginas PDF relevantes
- Síntesis de voz de las respuestas
- Exportación de conversaciones

================================================================================
REQUISITOS PREVIOS
================================================================================

1. Python 3.10 o superior
   - Descargar desde: https://www.python.org/downloads/

2. Node.js 18.x o superior
   - Descargar desde: https://nodejs.org/

3. Tesseract OCR (ya incluido en backend/Tesseract-OCR)
   - Si no funciona, descargar desde: https://github.com/UB-Mannheim/tesseract/wiki

4. Credenciales de Azure OpenAI
   - Necesarias para el archivo .env

================================================================================
INSTALACIÓN
================================================================================

-----------------------------------------
PASO 1: CONFIGURAR BACKEND (Python/FastAPI)
-----------------------------------------

1.1. Abrir terminal/PowerShell y navegar al directorio del backend:

     cd app_RAG-videos-docs\backend

1.2. Crear un entorno virtual de Python:

     python -m venv venv

1.3. Activar el entorno virtual:

     En Windows (PowerShell):
     .\venv\Scripts\Activate.ps1
     
     En Windows (CMD):
     .\venv\Scripts\activate.bat
     
     En Linux/Mac:
     source venv/bin/activate

1.4. Instalar las dependencias:

     pip install -r requirements.txt

     Nota: Si hay errores, actualizar pip primero:
     python -m pip install --upgrade pip

-----------------------------------------
PASO 2: CONFIGURAR FRONTEND (Next.js/React)
-----------------------------------------

2.1. Abrir una NUEVA terminal/PowerShell y navegar al directorio del frontend:

     cd app_RAG-videos-docs\frontend

2.2. Instalar las dependencias de Node.js:

     npm install

     Nota: Si hay errores, intentar con:
     npm install --legacy-peer-deps

-----------------------------------------
PASO 3: CONFIGURAR VARIABLES DE ENTORNO
-----------------------------------------

3.1. El archivo .env ya debe estar en la raíz del proyecto (app_RAG-videos-docs\.env)

3.2. Verificar que contenga las siguientes variables:

     AZURE_FOUNDRY_API_KEY=<tu-api-key>
     AZURE_FOUNDRY_API_KEY_gpt-4o=<tu-api-key>
     AZURE_TEXT_EMBEDDING_API_KEY=<tu-api-key>
     AZURE_OPENAI_ENDPOINT=<tu-endpoint>
     AZURE_OPENAI_WHISPER_DEPLOYMENT=whisper
     AZURE_OPENAI_TTS_DEPLOYMENT=gpt-4o-mini-tts
     AZURE_OPENAI_TTS_MODEL=gpt-4o-mini-tts
     AZURE_OPENAI_TTS_VOICE=alloy
     AZURE_OPENAI_API_VERSION=2025-03-01-preview
     
     # Deployment para OCR con visión (debe soportar imágenes)
     AZURE_OCR_DEPLOYMENT=gpt-4o
     AZURE_OPENAI_GPT4V_DEPLOYMENT=gpt-4o
     
     # Parámetros del procesamiento de video
     SCENE_DETECTION=false
     FRAME_EVERY_SECS=3
     MAX_FRAMES=20
     DO_OCR_FRAMES=false

3.3. Reemplazar los valores <tu-api-key> y <tu-endpoint> con tus credenciales de Azure

NOTA: DO_OCR_FRAMES controla si se realiza OCR en los frames del video.
      Por defecto está en 'false' porque el OCR con Azure es costoso.
      Puedes activarlo desde la interfaz (⚙️ Configuración) y guardarlo.

================================================================================
EJECUCIÓN
================================================================================

IMPORTANTE: Debes ejecutar tanto el backend como el frontend en terminales separadas.

-----------------------------------------
INICIAR BACKEND
-----------------------------------------

1. Abrir terminal/PowerShell en el directorio del backend:

   cd app_RAG-videos-docs\backend

2. Activar el entorno virtual (si no está activo):

   .\venv\Scripts\Activate.ps1

3. Iniciar el servidor backend:

   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

4. El backend estará disponible en:
   http://localhost:8000

   API Docs: http://localhost:8000/docs

-----------------------------------------
INICIAR FRONTEND
-----------------------------------------

1. Abrir una NUEVA terminal/PowerShell en el directorio del frontend:

   cd app_RAG-videos-docs\frontend

2. Iniciar el servidor de desarrollo:

   npm run dev

3. El frontend estará disponible en:
   http://localhost:3000

-----------------------------------------
ACCEDER A LA APLICACIÓN
-----------------------------------------

1. Abrir navegador web en: http://localhost:3000

2. Cargar un documento (PDF, DOCX, TXT) o video (MP4)

3. Esperar a que se procese e indexe el contenido

4. Comenzar a hacer preguntas en el chat

================================================================================
ESTRUCTURA DEL PROYECTO
================================================================================

app_RAG-videos-docs/
│
├── .env                          # Variables de entorno (credenciales Azure)
│
├── backend/                      # Servidor FastAPI (Python)
│   ├── venv/                     # Entorno virtual de Python (generado)
│   ├── main.py                   # API principal
│   ├── pdf_processor.py          # Procesador de PDFs
│   ├── requirements.txt          # Dependencias Python
│   ├── Tesseract-OCR/            # OCR embebido
│   ├── uploads/                  # Archivos cargados por usuarios
│   └── static/                   # Imágenes generadas de PDFs
│
└── frontend/                     # Aplicación Next.js (React)
    ├── app/                      # Páginas y componentes
    │   ├── page.js               # Página principal
    │   ├── layout.js             # Layout de la app
    │   └── globals.css           # Estilos globales
    ├── node_modules/             # Dependencias Node (generado)
    ├── package.json              # Dependencias y scripts
    └── Logo_Baufest_PNG.png      # Logo de la empresa

================================================================================
USO DE LA APLICACIÓN
================================================================================

1. CARGAR DOCUMENTO/VIDEO
   - Click en "Cargar documento/video"
   - Seleccionar archivo (PDF, DOCX, TXT o MP4)
   - Esperar a que se procese

2. CONSULTAR
   - Escribir pregunta en el chat
   - Presionar Enter o click en Enviar
   - Esperar respuesta del asistente

3. CONFIGURACIÓN DE PROCESAMIENTO DE VIDEO
   - Click en el ícono de configuración (⚙️)
   - Ajustar parámetros:
     * Frames cada X segundos
     * Máximo de frames a procesar
     * Detección de escenas
     * Realizar OCR en frames (extrae texto de las imágenes del video)
   - Click en "Guardar en .env" para persistir la configuración

4. FUNCIONES ADICIONALES
   - 🔊 Escuchar respuesta con síntesis de voz
   - 📄 Ver páginas del PDF relacionadas con la respuesta
   - 💾 Exportar conversación a TXT o DOCX
   - 🔄 Limpiar conversación

================================================================================
SOLUCIÓN DE PROBLEMAS
================================================================================

PROBLEMA: "uvicorn no se reconoce como comando"
SOLUCIÓN: Asegurarse de activar el entorno virtual del backend antes de ejecutar uvicorn

PROBLEMA: Error al instalar dependencias de Python
SOLUCIÓN: Actualizar pip: python -m pip install --upgrade pip

PROBLEMA: Error "Module not found" en el backend
SOLUCIÓN: Verificar que el entorno virtual esté activado y las dependencias instaladas

PROBLEMA: Frontend no se conecta al backend
SOLUCIÓN: Verificar que el backend esté corriendo en http://localhost:8000

PROBLEMA: Errores con Tesseract OCR
SOLUCIÓN: Verificar que exista backend/Tesseract-OCR/tesseract.exe

PROBLEMA: Errores con Azure OpenAI
SOLUCIÓN: Verificar credenciales en el archivo .env

================================================================================
COMANDOS RÁPIDOS
================================================================================

BACKEND:
--------
Activar entorno:   .\venv\Scripts\Activate.ps1
Instalar deps:     pip install -r requirements.txt
Ejecutar:          uvicorn main:app --host 0.0.0.0 --port 8000 --reload

FRONTEND:
---------
Instalar deps:     npm install
Ejecutar dev:      npm run dev
Build producción:  npm run build
Ejecutar prod:     npm start

================================================================================
Última actualización: Noviembre 7, 2025
