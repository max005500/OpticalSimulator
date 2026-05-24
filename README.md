# OpticalSimulator

Simulador óptico desarrollado como parte de mi proyecto de tesis, con el objetivo de generar datos sintéticos, analizarlos y utilizarlos como base para una futura implementación experimental o práctica.

Este repositorio contiene herramientas para modelar un sistema óptico, simular efectos atmosféricos, generar pantallas de fase y estudiar el comportamiento de sensores ópticos, especialmente en configuraciones relacionadas con sensores de frente de onda tipo Shack-Hartmann.

## 📌 Descripción

`OpticalSimulator` nace como una herramienta de apoyo para investigación y desarrollo en el contexto de una tesis.  
Su propósito principal es permitir la simulación de fenómenos ópticos y atmosféricos para generar datos controlados, analizarlos y extraer información útil antes de llevar el sistema a una implementación real.

El repositorio permite explorar distintos parámetros físicos y ópticos, tales como:

- diámetro de apertura;
- obstrucción central;
- tamaño de sensor;
- tamaño de píxel;
- número de lenslets;
- distancia focal;
- longitud de onda;
- turbulencia atmosférica;
- pantallas de fase tipo Von Kármán;
- propagación óptica;
- respuesta de sensores Shack-Hartmann.

## 🎯 Objetivo del proyecto

El objetivo principal del simulador es servir como entorno de prueba para:

1. Generar datos sintéticos bajo condiciones ópticas controladas.
2. Simular perturbaciones atmosféricas.
3. Analizar la respuesta de un sistema óptico frente a distintas configuraciones.
4. Estudiar sensores de frente de onda.
5. Validar ideas antes de llevarlas a una implementación experimental.
6. Apoyar el desarrollo y análisis asociado a mi proyecto de tesis.

## 📁 Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `OpticalSystem.py` | Define una clase para configurar el sistema óptico principal, incluyendo apertura, sensor, magnificación y configuración del sensor Shack-Hartmann. |
| `atm.py` | Contiene clases para modelar atmósfera local, capas atmosféricas y propagación con turbulencia. |
| `atmosfera.py` | Implementa herramientas para generar pantallas de fase Von Kármán, evolución temporal de turbulencia y transformaciones espaciales. |
| `shwfs.py` | Define componentes ópticos asociados a arreglos de microlentes y sensores Shack-Hartmann centrados o no centrados. |
| `Test.ipynb` | Notebook de pruebas y experimentación del simulador. |
| `Shimm.ipynb` | Notebook relacionado a simulacion utilizando HCIPy y otras librerias. |
| `LICENSE` | Licencia MIT del proyecto. |

## 🧠 Conceptos utilizados

Este proyecto trabaja con conceptos de óptica computacional y simulación física, entre ellos:

- óptica geométrica y física;
- propagación de frente de onda;
- sensores Shack-Hartmann;
- microlentes;
- pupilas ópticas;
- turbulencia atmosférica;
- pantallas de fase;
- modelo de Von Kármán;
- parámetro de Fried `r0`;
- escala externa `L0`;
- análisis de imágenes;
- generación de datos sintéticos.

## 🛠️ Tecnologías y librerías

El proyecto utiliza Python y notebooks como base de desarrollo.  
Entre las principales librerías utilizadas se encuentran:

- Python
- Jupyter Notebook
- NumPy
- SciPy
- Matplotlib
- PyTorch
- scikit-image
- HCIPy
- AOtools
- Rich

> Las dependencias pueden variar según el notebook o módulo utilizado.

## 🚀 Instalación básica

Clonar el repositorio:

```bash
git clone https://github.com/max005500/OpticalSimulator.git
cd OpticalSimulator
pip install -r requeriments.txt