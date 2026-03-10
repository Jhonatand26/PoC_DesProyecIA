# Variables
PYTHON = python3
PIP = pip
VENV = venv
MODEL_NAME = plantnet_model.pth

# 1. Configuración inicial y carpetas
setup:
	@echo "Creando estructura de directorios..."
	mkdir -p src/data_processing
	mkdir -p src/inference
	mkdir -p app/static
	mkdir -p models
	mkdir -p tests
	touch src/__init__.py src/inference/engine.py app/main.py
	@echo "Estructura creada con éxito."

# 2. Instalación de dependencias de HuggingFace y PyTorch
install:
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install transformers huggingface_hub pillow
	$(PIP) install streamlit  # Para la GUI rápida
	@echo "Librerías instaladas."

# 3. Script rápido para descargar el modelo inicial
download_model:
	$(PYTHON) -c "import torch; \
	model = torch.hub.load('prof-freakenstein/plantnet-disease-detection', 'model', trust_repo=True); \
	torch.save(model.state_dict(), 'models/$(MODEL_NAME)'); \
	print('Modelo guardado en models/')"

# 4. Limpieza del proyecto
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

# 5. Ejecutar la App (GUI)
run_app:
	streamlit run app/main.py

.PHONY: setup install download_model clean run_app
