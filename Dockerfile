FROM visheshpfm/insightface:1.4

# Trabajamos en la raíz, donde ya está handler.py
WORKDIR /

# Copiamos la API
COPY api.py /api.py


ENTRYPOINT []
# Instalamos FastAPI + Uvicorn
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

ENV PYTHONUNBUFFERED=1

# Exponemos el puerto de la API
EXPOSE 8000

# Arrancamos servidor FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

