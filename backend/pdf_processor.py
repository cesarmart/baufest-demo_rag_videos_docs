import fitz  # PyMuPDF
import os
import hashlib
from pathlib import Path

class PDFProcessor:
    def __init__(self, pdf_path, output_dir="static/pdf_pages"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.pdf_hash = self._generate_pdf_hash()
        self.page_images = {}
        
        # Crear directorio si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def _generate_pdf_hash(self):
        """Genera un hash único para el PDF para evitar regenerar imágenes si no ha cambiado"""
        with open(self.pdf_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash[:10]
    
    def extract_page_images(self, dpi=200):
        """Extrae imágenes de todas las páginas del PDF"""
        doc = fitz.open(self.pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Generar nombre de archivo para la imagen
            img_filename = f"{self.pdf_hash}_page_{page_num+1}.png"
            img_path = os.path.join(self.output_dir, img_filename)
            
            # Verificar si la imagen ya existe
            if not os.path.exists(img_path):
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                pix.save(img_path)
            
            # Guardar la ruta relativa
            self.page_images[page_num+1] = f"/static/pdf_pages/{img_filename}"
        
        return self.page_images
