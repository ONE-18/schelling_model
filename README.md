# Modelo de segregación de Schelling

Proyecto simple para simular el modelo de segregación de Schelling en dos formatos:

- **Python** con interfaz gráfica usando `tkinter`
- **HTML / JavaScript** para ejecución en navegador

## Archivos

- [schelling_segregation_model.py](schelling_segregation_model.py): versión en Python con controles interactivos.
- [schelling_segregation_model.html](schelling_segregation_model.html): versión web interactiva.

## ¿Qué hace la simulación?

El modelo coloca dos tipos de agentes en una cuadrícula, junto con celdas vacías. Cada agente evalúa si está “satisfecho” según la proporción de vecinos similares dentro de su vecindario. Si no alcanza el umbral de tolerancia, se mueve a una celda vacía. Con el tiempo, esto puede producir patrones de segregación incluso cuando la preferencia individual es moderada.

## Requisitos

### Versión Python

- Python 3.x
- `tkinter` incluido normalmente con la instalación estándar de Python en Windows

### Versión HTML

- Un navegador moderno

## Cómo ejecutar la versión en Python

En la carpeta del proyecto:

```bash
python schelling_segregation_model.py
```

## Controles disponibles

- **Tamaño de cuadrícula**: cambia el tamaño del tablero
- **% celdas vacías**: ajusta la proporción inicial de espacios vacíos
- **Umbral de tolerancia**: define el porcentaje mínimo de vecinos similares para que un agente esté satisfecho
- **Velocidad**: controla la rapidez de la simulación
- **Reiniciar**: genera una nueva distribución inicial
- **Paso**: avanza una generación
- **Ejecutar / Pausar**: inicia o detiene la simulación automática

## Estadísticas mostradas

- **Generación**: número de iteraciones transcurridas
- **Insatisfechos**: cantidad y porcentaje de agentes que desean moverse
- **Segregación**: índice aproximado basado en la similitud con vecinos

## Nota

La versión en Python reproduce la lógica principal del modelo interactivo del archivo HTML y está pensada para exploración educativa y visualización del comportamiento emergente.
