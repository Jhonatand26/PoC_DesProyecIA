# Gitflow — Estrategia de Ramas
## PoC Asistente Fitosanitario con IA | UAO 2026

---

## Estructura de Ramas

```
main
  └── develop
        └── feature/nombre-tarea
        └── feature/otro-issue
              ...
        └── release/v1.0  (solo al final del módulo)
```

| Rama | Propósito | Quién hace push |
|---|---|---|
| `main` | Producción — lo que se despliega en DigitalOcean | Solo desde `release/` vía PR |
| `develop` | Integración — donde todo el equipo converge | Solo desde `feature/` vía PR aprobado |
| `feature/xxx` | Trabajo individual por Issue | El responsable del Issue |
| `release/vX.X` | Preparación para entrega final | Jhonatan (arquitecto) |
| `hotfix/xxx` | Emergencias en producción | Jhonatan (arquitecto) |

---

## Reglas de Protección de Ramas

Las ramas `main` y `develop` están **protegidas en GitHub**:

- ❌ Ningún integrante puede hacer `git push` directo
- ✅ Todo cambio entra mediante Pull Request
- ✅ El PR requiere mínimo **1 aprobación** de otro integrante
- ✅ El PR debe pasar los checks de CI antes de mergear

---

## Ciclo de Vida de una Feature

### 1. Arrancar una tarea

```bash
# Siempre desde develop actualizado
git checkout develop
git pull origin develop
git checkout -b feature/nombre-tarea
```

### 2. Trabajar y commitear

```bash
# Commits pequeños y frecuentes referenciando el Issue
git add .
git commit -m "feat(modulo): descripcion breve #N"
```

### 3. Subir la rama

```bash
git push origin feature/nombre-tarea
# Primera vez:
git push -u origin feature/nombre-tarea
```

### 4. Abrir Pull Request

- **Base:** `develop`
- **Compare:** `feature/nombre-tarea`
- **Descripción:** incluir `Closes #N`
- **Reviewer:** un compañero del equipo

### 5. Code Review y Merge

- El reviewer tiene máximo **24 horas** para revisar
- Si hay cambios solicitados → el autor los corrige y pushea
- Si está aprobado → **Squash and merge** a `develop`
- El Issue se cierra automáticamente con `Closes #N`

---

## Convención de Nombres de Ramas

```bash
# Jhonatan — Backend & Arquitectura
feature/setup-uv
feature/setup-gitignore
feature/setup-gitflow
feature/grpc-proto-cv
feature/grpc-server-cv
feature/grpc-proto-nlp
feature/grpc-server-nlp
feature/dockerfile

# Jorge Luis — Visión Computacional
feature/cv-model-loader
feature/cv-preprocessor
feature/cv-classifier
feature/cv-tests

# Mateo — NLP y Gemini
feature/nlp-prompt-builder
feature/nlp-gemini-client
feature/nlp-prompt-validation
feature/nlp-tests

# Nicolas — Frontend y QA
feature/frontend-streamlit
feature/frontend-grpc-client
feature/qa-image-bank
feature/qa-ciclo
```

---

## Tareas con Dependencias entre Módulos

Cuando una tarea depende del trabajo de otro integrante, el flujo es:

```
1. El integrante A termina su feature → merge a develop
2. El integrante B hace: git checkout develop && git pull
3. El integrante B crea su feature desde develop
   → ya tiene el trabajo de A disponible
4. Si A hace un fix posterior → B sincroniza:
   git merge develop (dentro de su feature branch)
```

**develop es siempre el punto de sincronización del equipo.**

---

## Cierre del Módulo — Rama Release

```bash
# Jhonatan crea la rama release cuando todo está integrado en develop
git checkout develop
git pull origin develop
git checkout -b release/v1.0

# Ajustes finales, bump de versión en pyproject.toml
# Luego dos PRs:
# 1. release/v1.0 → main
# 2. release/v1.0 → develop
```

---

## Links del Proyecto

- **Repositorio:** github.com/Jhonatand26/PoC_DesProyecIA
- **Kanban:** github.com/users/Jhonatand26/projects/1
- **Rama activa:** `develop` — actualizar SIEMPRE antes de crear una `feature/`