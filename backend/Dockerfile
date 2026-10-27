FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install CPU-only PyTorch and torchvision
RUN python -m pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    && python -c "import torch; import torchvision; print('PyTorch:', torch.__version__)"

# Install FastAPI and other dependencies
COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

# Verify PyTorch is still installed
RUN python -c "import torch; import torchvision; print('PyTorch verified:', torch.__version__)"

# Copy application and model
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]