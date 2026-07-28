# Fuente canónica del generador sintético

## Decisión

La única distribución Python activa es:

- distribución: `synthetic-image-generator`
- import público: `synthetic_image_generator`
- versión: `0.2.0`
- fuente física: `igc6_universal_selfcontained/generators/`
- contrato del schedule: `sig-154-cycle-v1`

`pyproject.toml` mapea ese directorio directamente al namespace público. No
existe una copia en `src/` ni un segundo renderizador dentro del wheel.

## Consumidores activos

- `universal_programmatic_image_ae` declara la librería en `requirements.txt`
  y usa únicamente `import synthetic_image_generator`.
- Los adaptadores multimodales del repositorio padre apuntan explícitamente a
  esta misma fuente canónica para evitar que sus snapshots históricos hagan
  shadowing.

El AE tiene 154 etiquetas de contenido. Su ruta `raster_patch` es la etiqueta
155 del renderer y nunca se usa para pedir una imagen al generador.

## Inventario de archivos históricos

La auditoría del workspace del 2026-07-27 encontró:

- `invertible_generator/generators/synthetic_image_generator.py`: snapshot
  compatible usado por comandos históricos; no es una distribución Python
  ni lo importan los consumidores activos.
- `invertible_generator/syntetic_image_generator/`: release histórica
  autocontenida con el nombre antiguo; no contiene `pyproject.toml`.
- `structuredmatrix/experiments/compression/synthetic_image_generator*.py`:
  iteraciones congeladas de un experimento de compresión; no son paquetes ni
  dependencias del AE.

Se conservan para reproducibilidad. “Una sola fuente” significa una sola
implementación activa e instalable, no borrar evidencia histórica.

## Comprobación

Desde esta carpeta:

```bash
python scripts/audit_workspace_generator_sources.py
python -m pytest tests -q
```

La primera orden falla si aparece una segunda distribución con el mismo
nombre, si el AE vuelve a contener módulos locales que oculten la librería o
si sus imports dejan de usar el namespace público. También falla si un
consumidor activo del repositorio padre vuelve a importar los snapshots
históricos.
