# Usar una imagen base oficial de Python
FROM python:3.9-slim

# Configurar el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo requirements.txt al contenedor
COPY requirements.txt .

RUN pip install --upgrade pip

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código fuente al contenedor
COPY . .

# Exponer el puerto en el que se ejecuta la aplicación
EXPOSE 5000

# Comando para ejecutar la aplicación Flask
CMD ["python", "dentaigemini.py"]