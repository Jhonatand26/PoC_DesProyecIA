.PHONY: limpiar backend frontend arbol #Indica que estas no son archivos reales, sino tareas a ejecutar
limpiar:
	cls
	cls
	cls
	@echo "Limpiando el proyecto..."

visback:
	@echo "Iniciando y compilando el backend..."
	make limpiar 
	uv run python src/api/cv_server.py

nlpback:
	@echo "Iniciando y compilando el backend..."
	make limpiar 
	uv run python src/api/nlp_server.py

front:
	@echo "Iniciando y compilando el frontend..."
	make limpiar 
	uv run streamlit run src/app/app.py

mlflow:
	@echo "Iniciando MLflow..."
	make limpiar 
	uv run python scripts/mlflow_server.py


arbol:
	@echo "Mostrando el Arbol de directorios del proyecto..."
	make limpiar
	tree /f

docker:
	@echo "Levantando servicios con Docker Compose..."
	docker compose up --build

docker-down:
	@echo "Deteniendo servicios Docker..."
	docker compose down
