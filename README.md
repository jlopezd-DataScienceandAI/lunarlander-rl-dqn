# DQN para LunarLander-v3

Implementación de **Deep Q-Network (DQN)** con TensorFlow/Keras para resolver el entorno
[`LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/) de Gymnasium.
El agente aprende a aterrizar el módulo lunar de forma suave y precisa, alcanzando una
puntuación media superior a 200 puntos.

## Características

- **Replay buffer circular**: las transiciones antiguas se sobrescriben cuando la memoria se llena.
- **Red objetivo (target network)** con actualización periódica, para estabilizar los Q-targets.
- **Pérdida Huber** y **recorte de gradientes** (`clipnorm=1.0`) para un entrenamiento más estable.
- **Política epsilon-greedy** con decaimiento exponencial de la exploración.
- **Warm-up** del buffer con experiencias aleatorias antes de empezar a aprender.
- **Frecuencia de entrenamiento configurable** (`learn()` cada 4 pasos de entorno).
- Gráfica de la evolución del score con media móvil y guardado del modelo en formato `.keras`.

## El entorno

| | |
|---|---|
| **Observaciones** | 8 (posición x/y, velocidades, ángulo, velocidad angular, contacto de cada pata) |
| **Acciones** | 4 discretas (no hacer nada, motor izquierdo, motor principal, motor derecho) |
| **Recompensa** | Positiva por aterrizar correctamente; negativa por consumir combustible o estrellarse |
| **Objetivo** | Puntuación media ≥ 200 |

## Arquitectura y configuración

Red totalmente conectada con dos capas ocultas de 256 neuronas (ReLU, inicialización He) y
salida lineal con un Q-valor por acción. Optimizador Adam.

| Hiperparámetro | Valor |
|---|---|
| Learning rate | 6.3e-4 |
| Batch size | 128 |
| Gamma | 0.99 |
| Tamaño del replay buffer | 100 000 |
| Exploración (ε) | 1.0 → 0.05, decay 0.985 por episodio |
| Frecuencia de entrenamiento | cada 4 pasos de entorno |
| Actualización de la red objetivo | cada 500 pasos de entorno |
| Máximo de episodios | 1000 |
| Criterio de parada | media ≥ 230 en los últimos 20 episodios |

## Resultados

| Métrica | Valor |
|---|---|
| Episodios de entrenamiento | 195 |
| Pasos de entrenamiento | 80 001 |
| Tiempo de entrenamiento | ≈ 18 min |
| Score medio (últimos 20 episodios) | 233 |
| **Score medio en prueba (30 episodios, política greedy)** | **236** |
| Episodios de prueba con score ≥ 200 | 25 de 30 |

El notebook documenta además los experimentos que no funcionaron (redes más profundas,
distintos decaimientos de ε, actualizaciones de la red objetivo demasiado frecuentes,
batches pequeños con redes grandes) y por qué.

## Instalación y uso

```bash
git clone https://github.com/jlopezdi/lunarlander-rl-dqn.git
cd lunarlander-rl-dqn

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook dqn_lunarlander.ipynb
```

Ejecuta las celdas en orden: el notebook entrena al agente, muestra la gráfica de
entrenamiento, guarda el modelo en `my_model.keras` y evalúa la política aprendida sobre
30 episodios.

> **Nota:** `box2d` necesita compilar código nativo y puede requerir
> [SWIG](https://www.swig.org/) instalado en el sistema. Con Conda:
> `conda install swig`.

## Estructura del repositorio

```
.
├── dqn_lunarlander.ipynb   # Implementación, entrenamiento, evaluación y análisis
├── requirements.txt        # Dependencias
├── .gitignore
└── README.md
```

## Autor

Jorge López · jorgelpdz05@gmail.com

## Referencias

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature 518.
- [Documentación de Gymnasium – LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
