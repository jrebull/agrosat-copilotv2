"""El marco de conjuntos: propiedades que tienen que cumplirse, y la relacion con lo anterior."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import recall_score

from ml.eval.set_valued import (
    SetPrediction,
    abstencion,
    coste_esperado,
    recorte,
    singleton,
    utilidad_macro,
)

#: Utilidad que solo premia el singleton correcto. Es la parametrizacion que conecta el marco
#: nuevo con el anterior, y sirve para comprobar que no estamos midiendo otra cosa sin saberlo.
G_SINGLETON = np.vectorize(lambda k: 1.0 if k == 1 else 0.0, otypes=[float])


@pytest.fixture
def universo() -> tuple[np.ndarray, np.ndarray]:
    """A small deterministic universe with an imbalanced label distribution."""
    rng = np.random.default_rng(0)
    n, c = 600, 6
    labels = rng.choice(c, size=n, p=[0.5, 0.2, 0.12, 0.1, 0.05, 0.03])
    proba = rng.dirichlet(np.ones(c) * 0.4, size=n)
    aciertos = rng.random(n) < 0.7
    proba[aciertos] = 0.02
    proba[np.flatnonzero(aciertos), labels[aciertos]] = 0.9
    return labels, proba / proba.sum(axis=1, keepdims=True)


def test_el_vacio_es_abstencion_y_no_un_caso_especial(universo) -> None:
    """Abstention is the empty set, so it lives in the same object as every other mechanism."""
    _, proba = universo
    pred = abstencion(proba, umbral=0.99)
    assert pred.vacios.any(), "con un umbral casi imposible tiene que abstenerse en algo"
    assert set(np.unique(pred.tamanos)) <= {0, 1}


def test_el_coste_esta_definido_para_todos_los_mecanismos(universo) -> None:
    """The cost axis exists for every mechanism, which is the whole point of the frame."""
    _, proba = universo
    legend = [0, 1, 2]
    for pred in (
        singleton(proba),
        recorte(proba, legend, "recorte"),
        abstencion(proba, umbral=0.5),
    ):
        coste = coste_esperado(pred, lambda k: k.astype(float))
        assert 0.0 <= coste <= proba.shape[1]
    # El predictor intacto entrega exactamente una clase por parcela.
    assert coste_esperado(singleton(proba), lambda k: k.astype(float)) == pytest.approx(1.0)


def test_la_utilidad_singleton_es_el_recall_macro_no_el_f1(universo) -> None:
    """With the singleton utility the estimand is macro RECALL, not macro F1.

    Esto no es un detalle: el marco nuevo NO es una reparametrizacion del anterior. Al evaluar
    sobre la poblacion completa en vez de sobre la entregada, una parcela no entregada cuenta como
    fallo en lugar de desaparecer del denominador, y eso es exactamente lo que hace que la metrica
    deje de premiar abstenerse donde mas pesa.
    """
    labels, proba = universo
    pred = singleton(proba)
    clases = sorted(set(labels.tolist()))
    esperado = recall_score(labels, proba.argmax(axis=1), labels=clases, average="macro")
    obtenido = utilidad_macro(labels, pred, G_SINGLETON, utilidad_abstencion=0.0, clases=clases)
    assert obtenido == pytest.approx(esperado, abs=1e-12)


def test_abstenerse_nunca_sube_la_utilidad_bajo_la_g_singleton(universo) -> None:
    """Abstention cannot buy utility when the whole population is scored.

    Es la propiedad que el estimando anterior NO tenia: alli abstenerse en las parcelas dificiles
    subia la macro porque desaparecian del denominador de su clase.
    """
    labels, proba = universo
    clases = sorted(set(labels.tolist()))
    completo = utilidad_macro(
        labels, singleton(proba), G_SINGLETON, utilidad_abstencion=0.0, clases=clases
    )
    for umbral in (0.3, 0.5, 0.7, 0.9):
        parcial = utilidad_macro(
            labels, abstencion(proba, umbral), G_SINGLETON, utilidad_abstencion=0.0, clases=clases
        )
        assert parcial <= completo + 1e-12


def test_el_recorte_no_lee_la_etiqueta(universo) -> None:
    """Delivery depends on the free argmax, never on the truth."""
    labels, proba = universo
    pred = recorte(proba, [0, 1], "recorte")
    esperado = np.isin(proba.argmax(axis=1), [0, 1])
    assert np.array_equal(~pred.vacios, esperado)
    # Cambiar las etiquetas no puede cambiar lo que se entrega.
    otras = (labels + 1) % proba.shape[1]
    assert np.array_equal(recorte(proba, [0, 1], "recorte").mask, pred.mask)
    assert otras is not labels


def test_sin_clases_sobre_el_suelo_falla_en_vez_de_devolver_cero(universo) -> None:
    """An undefined estimand raises; it does not return a valid-looking zero."""
    labels, proba = universo
    with pytest.raises(ValueError, match="suelo de soporte"):
        utilidad_macro(
            labels, singleton(proba), G_SINGLETON, utilidad_abstencion=0.0, soporte_minimo=10_000
        )


def test_la_mascara_entera_se_rechaza(universo) -> None:
    """An integer mask is a silent bug waiting to happen, so it is refused."""
    _, proba = universo
    with pytest.raises(TypeError, match="booleana"):
        SetPrediction(np.zeros(proba.shape, dtype=int), "mala")


def test_el_precio_de_abstenerse_se_puede_elegir_de_verdad(universo) -> None:
    """The declared price of abstention must actually move the estimand.

    Regresion de un fallo real: la primera version multiplicaba la utilidad por la contencion, con
    lo que el conjunto vacio valia cero pasara lo que pasara y `g(0)` no se evaluaba nunca. El
    modulo prometia dejar elegir cuanto vale abstenerse y lo hacia imposible. Y el test anterior no
    podia detectarlo porque usaba el unico valor que no distingue, que es cero: el mismo modo de
    fallo que un gate de anonimato ciego a sus propios acentos.
    """
    labels, proba = universo
    clases = sorted(set(labels.tolist()))
    pred = abstencion(proba, umbral=0.5)
    assert pred.vacios.any(), "el caso solo tiene sentido si de verdad se abstiene en algo"
    valores = [
        utilidad_macro(labels, pred, G_SINGLETON, utilidad_abstencion=u, clases=clases)
        for u in (0.0, 0.5, 1.0)
    ]
    assert valores[0] < valores[1] < valores[2], (
        f"el precio de abstenerse no mueve el estimando: {valores}"
    )


def test_abstenerse_gratis_hace_que_abstenerse_convenga(universo) -> None:
    """A pathological price makes abstention dominate, which is how we know it is priced at all.

    No es un caso que vayamos a usar: es la prueba de que la palanca existe. Si con
    `utilidad_abstencion = 1` abstenerse en todo no gana, es que el termino no esta conectado.
    """
    labels, proba = universo
    clases = sorted(set(labels.tolist()))
    todo = utilidad_macro(
        labels, abstencion(proba, umbral=2.0), G_SINGLETON, utilidad_abstencion=1.0, clases=clases
    )
    nada = utilidad_macro(
        labels, singleton(proba), G_SINGLETON, utilidad_abstencion=1.0, clases=clases
    )
    assert todo == pytest.approx(1.0), "abstenerse siempre con precio uno tiene que valer uno"
    assert todo > nada
