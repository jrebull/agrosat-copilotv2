"""Shared prose for the US-030..US-040 descriptive notebooks (06a/06b/06c).

Centralizes the glossary, the acronym table and the per-US narrative blocks so
the three build scripts stay DRY and the numbers (all from real PASTIS-R runs)
live in one place. The prose is neutral Spanish for the reader; identifiers and
keys stay English ASCII (project rule).

The glossary and acronym table are written so the FIRST notebook a reader opens
explains every difficult term and every abbreviation, per the explicit request:
"si usas terminos dificiles agrega un glosario y si abrevias o usas siglas
tambien quiero que expliques en la primera que es".
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Acronyms / abbreviations -- explained on first use (shown in every notebook).
# ---------------------------------------------------------------------------
ACRONYMS: list[tuple[str, str]] = [
    ("US", "User Story (historia de usuario): una unidad de trabajo del plan SCRUM."),
    ("EPIC", "Conjunto de User Stories relacionadas (aqui E5 modelos / E6 ensambles)."),
    (
        "PASTIS-R",
        "Panoptic Agricultural Satellite TIme Series (version Radar). Dataset "
        "publico frances de series Sentinel-1/2 con etiquetas de cultivo por pixel.",
    ),
    ("S2", "Sentinel-2: satelite optico de la ESA (10 bandas usadas aqui)."),
    ("S1", "Sentinel-1: satelite radar (SAR) de la ESA."),
    ("RGB", "Red-Green-Blue: las 3 bandas visibles que forman la imagen a color."),
    (
        "NDVI",
        "Normalized Difference Vegetation Index: indice de vigor de la vegetacion "
        "calculado con la banda roja y la infrarroja cercana.",
    ),
    ("NIR", "Near-InfraRed (infrarrojo cercano): banda muy sensible a la vegetacion."),
    ("GT", "Ground Truth (verdad de campo): la etiqueta correcta de referencia."),
    (
        "CLS",
        "Token de Clasificacion (Class token): el vector resumen de toda la imagen "
        "que produce un Vision Transformer.",
    ),
    (
        "mIoU",
        "mean Intersection over Union: promedio del solapamiento entre prediccion "
        "y verdad por clase (0 = nada, 1 = perfecto). Metrica estandar de segmentacion.",
    ),
    ("IoU", "Intersection over Union: solapamiento prediccion/verdad de una sola clase."),
    (
        "F1-macro",
        "Promedio simple del F1 (media armonica de precision y exhaustividad) "
        "sobre todas las clases; trata igual a clases raras y comunes.",
    ),
    (
        "OOF",
        "Out-Of-Fold: predicciones hechas sobre datos que el modelo NO vio al "
        "entrenar (evita el auto-engano / fuga de informacion).",
    ),
    ("CV", "Cross-Validation (validacion cruzada): partir el dato en folds para evaluar."),
    ("fold", "Particion del dataset. PASTIS trae 5 folds espacialmente separados."),
    ("held-out", "Fold reservado, nunca usado en entrenamiento, para la evaluacion final."),
    ("FM", "Foundation Model (modelo fundacional): modelo grande pre-entrenado y reusable."),
    ("VLM", "Vision-Language Model: modelo que entiende imagen y texto a la vez."),
    ("LLM", "Large Language Model (modelo grande de lenguaje)."),
    (
        "MPCL",
        "Multi-Positive Contrastive Loss: perdida contrastiva que agrupa varios "
        "ejemplos positivos de la misma categoria (nucleo del FarSLIP fiel).",
    ),
    (
        "L_glo / L_loc",
        "Perdidas global (imagen<->caption) y local (region<->categoria) del paper FarSLIP.",
    ),
    ("LoRA", "Low-Rank Adaptation: tecnica para afinar modelos grandes con pocos pesos."),
    ("H100", "GPU NVIDIA H100 (96 GB) usada para el entrenamiento real del lote."),
    ("DVC", "Data Version Control: versiona datos/pesos grandes fuera de Git (en GCS)."),
    ("GCS", "Google Cloud Storage: el almacenamiento remoto de los artefactos DVC."),
    ("MLflow", "Herramienta de tracking de experimentos (metricas, parametros, lineage)."),
    ("CM", "Confusion Matrix (matriz de confusion): tabla verdad vs prediccion por clase."),
    ("T_MIN", "Numero minimo de fechas de una serie temporal en PASTIS (= 37)."),
    (
        "Full-M",
        "Configuracion 'Medium' completa del TSViT (dim 192, 6+6 capas), la del "
        "paper Tarasiou 2023, habilitada por la H100.",
    ),
]

# ---------------------------------------------------------------------------
# Glossary -- difficult concepts explained in plain language.
# ---------------------------------------------------------------------------
GLOSSARY: list[tuple[str, str]] = [
    (
        "Segmentacion semantica",
        "Asignar una clase de cultivo a CADA pixel de la imagen (no una etiqueta por "
        "imagen). El resultado es un mapa coloreado del mismo tamano que la entrada.",
    ),
    (
        "Clasificacion por patch",
        "Asignar UNA sola etiqueta a todo el recorte (patch). Es lo que hace FarSLIP: "
        "mas debil que la segmentacion porque no distingue parcelas dentro del patch.",
    ),
    (
        "Parcela / region",
        "Una porcion contigua de terreno con un solo cultivo (monocultivo). Un patch "
        "de PASTIS contiene muchas parcelas de distintos cultivos.",
    ),
    ("Patch", "Recorte cuadrado de 128x128 pixeles del satelite. La unidad basica del dataset."),
    (
        "Embedding",
        "Vector numerico que resume el contenido de una imagen o una parcela; modelos "
        "parecidos quedan cerca en ese espacio.",
    ),
    (
        "AlphaEarth",
        "Modelo fundacional de Google (Satellite Embedding V1) que entrega un embedding "
        "de 64 dimensiones por celda, entrenado con datos globales masivos.",
    ),
    (
        "FarSLIP fiel",
        "Reimplementacion fiel al paper (Li et al. 2025) del afinado de un modelo CLIP "
        "usando pares region-categoria y captions; busca alinear imagen y texto agricola.",
    ),
    (
        "Caption",
        "Descripcion en lenguaje natural de la imagen, generada aqui por Gemma 4 "
        "multimodal, que alimenta la perdida global L_glo del FarSLIP.",
    ),
    (
        "Ensamble",
        "Combinacion de varios modelos para superar al mejor individual (votacion, "
        "bagging, stacking, blending).",
    ),
    (
        "Fuga de informacion (leakage)",
        "Cuando el modelo ve, directa o indirectamente, datos de evaluacion al entrenar; "
        "infla las metricas y las vuelve mentirosas. Todo el lote la evita por diseno.",
    ),
    (
        "Tabla apples-to-apples",
        "Comparacion en condiciones identicas (mismo fold, misma metrica, mismo esquema "
        "de clases) para que los numeros sean realmente comparables.",
    ),
    (
        "Techo del dataset",
        "El maximo que cualquier modelo puede alcanzar dado el dato disponible; en PASTIS "
        "con FarSLIP resultan ~4 clases bien resueltas, no por fallo del metodo sino por "
        "el limite del dato.",
    ),
    (
        "1-CLS-por-patch",
        "Limitacion clave del FarSLIP: como cada patch produce UN solo vector resumen, "
        "todas las parcelas del patch reciben la misma prediccion; una parcela rara "
        "dentro de un patch de pradera es irrecuperable.",
    ),
]

# US one-line descriptions for the index cell (real, from docs/us-resolved).
US_ONE_LINERS: dict[str, str] = {
    "US-030": "Harness unico de re-score de los 6 segmentadores en fold-5 held-out "
    "(condiciones apples-to-apples, 18 clases).",
    "US-031": "Volcado de probabilidades OOF (softmax por pixel) de los 6 segmentadores "
    "en fold-5, insumo de los ensambles.",
    "US-032": "Filtro 3:1 de dominancia de Meadow por patch (modo nuevo del PastisFilter).",
    "US-033": "Prototipos de fenologia reales por clase (curva NDVI -> texto Gemini -> "
    "embedding MiniLM 384).",
    "US-034": "Fix critico: reemplazar prototipos aleatorios (torch.randn) por los "
    "fenologicos reales en la perdida contrastiva de FarSLIP.",
    "US-035": "Ablacion de bandas FarSLIP (rgb / nir_rgb / 4band) entrenada en H100.",
    "US-036-a-v2": "FarSLIP fiel al paper (region-categoria, MPCL + L_glo) sobre PASTIS "
    "real; techo del dataset = ~4 clases.",
    "US-037": "Comparativa apples-to-apples FarSLIP fiel vs AlphaEarth en separabilidad "
    "(gana AlphaEarth: F1 0.645 vs 0.555).",
    "US-038": "Re-entreno TSViT con la config Full-M completa en H100 "
    "(val_miou fold-4 0.699; mejor segmentador individual).",
    "US-039": "Re-entreno TSViT-pheno Full-M (ablacion honesta): la rama fenologica no "
    "aporta margen en supervisado full-label (saturacion).",
    "US-040": "Cuatro ensambles base (Voting/Bagging/Stacking/Blending): Stacking 0.747 "
    "elegido, supera al mejor individual.",
}
