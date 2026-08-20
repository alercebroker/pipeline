# Feature step — cambios TEMPORALES para correr local (arm64 / Apple Silicon)

> ⚠️ **Todos los cambios de este doc son SOLO para correr el step localmente en macOS
> Apple Silicon.** La imagen Docker se construye en **Linux x86-64**, donde el código
> original funciona tal cual. **Antes de construir la imagen hay que revertir todo esto.**

Fecha: 2026-07-09 · Branch: `features_step`

---

## Por qué existen estos cambios

1. **`-march=x86-64-v3`**: `P4J` y `mhps` fuerzan ese flag de compilador (Intel) en su
   `setup.py`. En arm64 clang lo rechaza (`unsupported argument 'x86-64-v3'`).
2. **`fastavro==1.6.1`**: `apf` lo capaba en `<=1.6.1`, y esa versión solo tiene wheel
   `macosx_x86_64` (no arm64) → intenta compilar de fuente y el Cython moderno rompe.
3. **Envío a scribe**: comentado para poder correr sin escribir en los tópicos reales.

En la imagen Linux x86-64 nada de esto aplica: el flag Intel es válido y fastavro 1.6.1
tiene wheel. Por eso **deben salir** antes del build.

---

## Inventario de cambios

### A. Ediciones intencionales (código/config)

| Archivo | Cambio |
|---|---|
| `P4J/setup.py` | `import platform` + omitir `-march=x86-64-v3` en arm64/aarch64 |
| `mhps/setup.py` | idem (arch-aware compiler flags) |
| `libs/apf/pyproject.toml` | `fastavro = ">=0.22.0,<=1.6.1"` → `">=0.22.0,<2"` |
| `feature_step/pyproject.toml` | agregado `fastavro = ">=1.9.5,<2"` (fuerza wheel arm64) |
| `feature_step/poetry.lock` | regenerado (fastavro → 1.12.2, entre otros) |
| `feature_step/features/step.py` | comentados `produce_to_scribe(...)` y `produce_xmatch_to_scribe(...)` (marcados con `# TEMP`) |

### B. Artefactos regenerados por el build de Cython (no editados a mano)

| Archivo | Origen |
|---|---|
| `P4J/P4J/algorithms/*.c` y `*.html` (12 archivos) | los reescribió `cythonize` al compilar |
| `mhps/mhps/mhps_wrapper.c` | idem |

### C. Archivo nuevo (local, NO commitear)

| Archivo | Nota |
|---|---|
| `feature_step/local_config.yaml` | config local con credenciales; mantener fuera de git |

---

## Cómo revertir (antes de construir la imagen)

Desde la raíz del repo (`/Users/panchoandrades/Desktop/repos/pipeline`):

```bash
# A + B: restaurar todo lo tracked a su estado original
git restore \
  P4J/setup.py \
  mhps/setup.py \
  libs/apf/pyproject.toml \
  feature_step/pyproject.toml \
  feature_step/poetry.lock \
  feature_step/features/step.py \
  P4J/P4J/algorithms \
  mhps/mhps/mhps_wrapper.c

# C: el config local es untracked; se saca solo si quieres
rm feature_step/local_config.yaml   # opcional
```

Verificar que quedó limpio (no deberían aparecer estos archivos):

```bash
git status --short
```

> Si en vez de revertir prefieres construir la imagen desde un checkout limpio de
> `main`/`features_step` (sin tu working copy sucio), también sirve — el Dockerfile
> parte del código commiteado, no de estos cambios locales.

---

## ⚠️ Lo más importante de no olvidar

- **`feature_step/features/step.py`**: si esto NO se revierte, la imagen de producción
  **dejaría de escribir a scribe**. Es el cambio más peligroso de olvidar.
- `libs/apf/pyproject.toml` y los dos `poetry.lock`/`pyproject.toml`: revertir para no
  arrastrar el bump de fastavro a producción sin haberlo validado.

---

## Estado del entorno local (no toca el repo, no requiere revert)

- Env de trabajo: virtualenv gestionado por poetry, basado en el `3.10.16` de pyenv:
  `~/Library/Caches/pypoetry/virtualenvs/feature-step-*-py3.10`
- Correr el step:
  ```bash
  cd feature_step
  poetry run env CONFIG_FROM_YAML=yes CONFIG_YAML_PATH=$(pwd)/local_config.yaml \
    python scripts/run_step.py
  ```
- Nota: durante el proceso también se instalaron paquetes en el Python **base**
  `~/.pyenv/versions/3.10.16` (por un intento con `virtualenvs.create=false`).
  No afecta al repo ni a la imagen; si quieres limpiar el base es aparte.
