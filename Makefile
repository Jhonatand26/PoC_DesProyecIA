.PHONY: limpiar backend frontend arbol #Indica que estas no son archivos reales, sino tareas a ejecutar
limpiar:
	cls
	cls
	cls
	@echo "Limpiando el proyecto..."

backend:
	@echo "Iniciando y compilando el backend..."
	make limpiar 
	pendiente por incluir todo lo del backend 

frontend:
	@echo "Iniciando y compilando el frontend..."
	make limpiar 
	pendiente por incluir todo lo del frontend

arbol:
	@echo "Mostrando el Arbol de directorios del proyecto..."
	make limpiar
	tree /f

docker: 
	@echo "Construyendo la imagen de Docker..."
	pendiente poner lo de doker
