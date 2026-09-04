"""Los mecanismos como predictores con valores de conjunto, sobre la poblacion completa.

Este modulo existe porque «a igual cobertura» **no esta definido** para dos de los cuatro
mecanismos que el articulo compara. Un conjunto conforme y una clase gruesa se entregan en el cien
por cien de las parcelas: su cobertura es una constante igual a uno, asi que igualarla no iguala
nada. Y el estimando anterior condicionaba sobre las parcelas entregadas, que es una variable
POSTERIOR al tratamiento, de modo que cada brazo se puntuaba sobre una poblacion distinta.

La solucion no es un parche sino el marco que la literatura ya tiene desde Ha (1997): expresar cada
mecanismo como una funcion que asigna a cada parcela un **subconjunto** del espacio de etiquetas.

    sin mecanismo        C(x) = {argmax}                       siempre un singleton
    recorte por F1       C(x) = {argmax en L} si argmax en L, si no el vacio
    recorte por soporte  igual, con L elegida por soporte
    abstencion           C(x) = {argmax} si la confianza pasa el umbral, si no el vacio
    conforme             C(x) = conjunto arbitrario
    retroceso jerarquico C(x) = las hojas bajo el nodo emitido

Con eso los seis viven en el mismo objeto y se puntuan sobre la poblacion COMPLETA de prueba, no
sobre la entregada.

**La cardinalidad NO es el coste, y este modulo ya no finge que lo sea.** Una auditoria externa lo
puso asi: `E[|C(x)|]` trata igual a dos conjuntos del mismo tamano aunque uno sea agronomicamente
inutil, y le pone precio cero al conjunto vacio. La moneda comun del articulo es una **tabla de
perdidas por accion, resultado y afectado**, y esa tabla no se deduce del dato: la fijan los casos
de uso (US-172). Hasta que exista, aqui solo vive `cardinalidad_esperada`, que es un **descriptor**
y se llama como lo que es. `g` sigue siendo la utilidad declarada de acertar con un conjunto de un
tamano dado, y solo eso.

**El valor de no responder es un termino aparte, y tiene que serlo.** Una auditoria externa
encontro que la primera version de este modulo multiplicaba la utilidad por la contencion, con lo
que el conjunto vacio caia a cero pasara lo que pasara y `g(0)` no se evaluaba nunca. Es decir: el
modulo prometia dejar elegir cuanto vale abstenerse y hacia imposible elegirlo. Por eso ahora

    u(y, C) = g(|C|) * 1[y en C]   si C no es vacio
    u(y, vacio) = utilidad_abstencion,  declarada aparte

**Elegir `utilidad_abstencion` ES la decision etica del articulo**, y hasta ahora estaba incrustada
en el F1-macro a un precio que nadie habia elegido. Se declara en el preregistro antes de mirar.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "NUM_CLASSES",
    "SetPrediction",
    "abstencion",
    "cardinalidad_esperada",
    "recorte",
    "singleton",
    "utilidad_macro",
]

#: Clases del universo primario.
NUM_CLASSES: int = 18


@dataclass(frozen=True)
class SetPrediction:
    """A set-valued prediction over a fixed universe of parcels.

    Attributes:
        mask: Boolean matrix ``(n_parcels, n_classes)``; ``mask[i, c]`` says whether class ``c``
            belongs to the set emitted for parcel ``i``. The empty set is a row of ``False`` and is
            how abstention is expressed, so every mechanism lives in the same object.
        mecanismo: Name recorded on every measurement taken from this prediction.
    """

    mask: np.ndarray
    mecanismo: str

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            raise TypeError("la mascara tiene que ser booleana, no un entero disfrazado")
        if self.mask.ndim != 2:
            raise ValueError(f"se esperaba (parcelas, clases) y llego {self.mask.shape}")

    @property
    def tamanos(self) -> np.ndarray:
        """Size of the set emitted for each parcel."""
        sizes: np.ndarray = self.mask.sum(axis=1)
        return sizes

    @property
    def vacios(self) -> np.ndarray:
        """Boolean mask of the parcels where nothing was emitted."""
        vacia: np.ndarray = self.tamanos == 0
        return vacia


def singleton(proba: np.ndarray, mecanismo: str = "sin mecanismo") -> SetPrediction:
    """The untouched predictor: one class per parcel, always.

    Args:
        proba: Posterior matrix ``(n_parcels, n_classes)``.
        mecanismo: Name to record.

    Returns:
        A prediction whose every set is a singleton.
    """
    mask = np.zeros(proba.shape, dtype=bool)
    mask[np.arange(proba.shape[0]), proba.argmax(axis=1)] = True
    return SetPrediction(mask, mecanismo)


def recorte(proba: np.ndarray, legend: Sequence[int], mecanismo: str) -> SetPrediction:
    """Legend shrinking, expressed as a set-valued predictor.

    The product promises only ``legend``. A parcel whose unrestricted argmax falls inside it gets
    the restricted argmax; one whose argmax falls outside gets the empty set. The delivery rule
    never reads a ground-truth label, which the retired implementation did.

    Args:
        proba: Posterior matrix.
        legend: Classes the product is allowed to emit.
        mecanismo: Name to record.

    Returns:
        The set-valued prediction of the mechanism.
    """
    columnas = np.asarray(sorted(legend), dtype=int)
    libre = proba.argmax(axis=1)
    dentro = np.isin(libre, columnas)
    emitido = columnas[proba[:, columnas].argmax(axis=1)]
    mask = np.zeros(proba.shape, dtype=bool)
    mask[np.flatnonzero(dentro), emitido[dentro]] = True
    return SetPrediction(mask, mecanismo)


def abstencion(proba: np.ndarray, umbral: float, mecanismo: str = "abstencion") -> SetPrediction:
    """Confidence rejection, expressed as a set-valued predictor.

    Args:
        proba: Posterior matrix.
        umbral: Confidence floor. It must come from the training blocks, never from the block being
            evaluated: choosing it inside the evaluated block is the asymmetry the audit found.
        mecanismo: Name to record.

    Returns:
        The set-valued prediction: a singleton where confident, the empty set elsewhere.
    """
    entrega = proba.max(axis=1) >= umbral
    mask = np.zeros(proba.shape, dtype=bool)
    filas = np.flatnonzero(entrega)
    mask[filas, proba[filas].argmax(axis=1)] = True
    return SetPrediction(mask, mecanismo)


def cardinalidad_esperada(pred: SetPrediction) -> float:
    """Mean size of the emitted set, counting abstention as zero.

    This is a **descriptor**, not the cost. It used to be called ``coste_esperado`` and to take a
    cost function, which is how cardinality ended up standing in for a loss nobody had declared: two
    sets of the same size are not equally expensive, and an empty set is not free. The cost axis is
    the loss table of US-172 and it does not live in this module.

    Args:
        pred: The set-valued prediction.

    Returns:
        Mean cardinality per parcel over the whole population.
    """
    return float(pred.tamanos.mean())


def utilidad_macro(
    labels: np.ndarray,
    pred: SetPrediction,
    g: Callable[[np.ndarray], np.ndarray],
    *,
    utilidad_abstencion: float,
    clases: Sequence[int] | None = None,
    soporte_minimo: int = 20,
) -> float:
    """Utility averaged per true class and then across classes.

    Two properties matter and both are deliberate. It is computed over the **whole** test
    population, so it does not condition on delivery, which is a post-treatment variable. And it
    averages per true class first, which keeps the macro spirit and connects straight to the
    accounting of who pays.

    Args:
        labels: Ground-truth labels.
        pred: The set-valued prediction.
        g: Declared utility of a NON-EMPTY set of a given size when the truth is inside it. It is
            never called with a zero: a function defined only for positive sizes is legitimate here
            and must not break.
        utilidad_abstencion: Declared utility of emitting nothing. It is a separate term on
            purpose: multiplying by containment, as the first version did, makes the empty set
            worth zero whatever the declaration says, so the price of abstaining could not be
            chosen. It has no default, because a default here is a decision taken by omission.
        clases: Classes to average over. Defaults to those meeting the support floor.
        soporte_minimo: Parcels a class needs to enter the average. Measured in the design study:
            below this the per-class estimate is noise, and at every block count some block holds a
            single parcel of some class.

    Returns:
        The macro utility.

    Raises:
        ValueError: if no class meets the support floor, which must never pass silently.
    """
    acierta = pred.mask[np.arange(labels.size), labels]
    vacios = pred.vacios
    valor = np.zeros(labels.size, dtype=float)
    # `g` se evalua SOLO sobre conjuntos no vacios. Calcularla sobre todos los tamanos y descartar
    # las columnas despues es el mismo error de antes en otra forma: la firma promete que g nunca ve
    # un cero, y una g legitima que rechace el cero reventaba igual.
    premiados = np.flatnonzero(acierta & ~vacios)
    if premiados.size:
        valor[premiados] = g(pred.tamanos[premiados])
    valor[vacios] = utilidad_abstencion
    if clases is None:
        cuenta = np.bincount(labels, minlength=pred.mask.shape[1])
        clases = [c for c in range(pred.mask.shape[1]) if cuenta[c] >= soporte_minimo]
    if not list(clases):
        raise ValueError(
            f"ninguna clase llega al suelo de soporte de {soporte_minimo}; "
            "el estimando no esta definido y no se devuelve un cero silencioso"
        )
    return float(np.mean([valor[labels == c].mean() for c in clases]))
