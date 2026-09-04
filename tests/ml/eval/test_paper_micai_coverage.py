"""Los tres defectos del aparato de la frontera, cada uno con un test que los detecta.

Este modulo no tenia tests. Los tres defectos que se le diagnosticaron sobrevivieron a una
primera reparacion y los encontro una auditoria externa **leyendo el codigo**, no ejecutandolo:
la suite verde no decia nada de este fichero. Cada test de aqui existe para que el defecto no
pueda volver en silencio, y cada uno se comprobo fallando sobre la implementacion anterior.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.eval.paper_micai_coverage import (
    BlockPoint,
    confidence_baseline,
    frontier,
    legend_by_f1,
    macro_over,
    paired_interval,
    presentes_en_bloque,
    umbral_desde_entrenamiento,
)


@pytest.fixture
def universo() -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """A small universe with four spatial blocks and a long-tailed label distribution."""
    rng = np.random.default_rng(7)
    n, c = 1200, 6
    labels = rng.choice(c, size=n, p=[0.45, 0.22, 0.15, 0.1, 0.05, 0.03])
    proba = rng.dirichlet(np.ones(c) * 0.5, size=n)
    aciertan = rng.random(n) < 0.65
    proba[aciertan] = 0.03
    proba[np.flatnonzero(aciertan), labels[aciertan]] = 0.85
    proba = proba / proba.sum(axis=1, keepdims=True)
    orden = rng.permutation(n)
    trozos = np.array_split(orden, 4)
    splits = [
        (np.setdiff1d(orden, t), np.asarray(sorted(t)))
        for t in (np.asarray(sorted(x)) for x in trozos)
    ]
    return proba, labels, splits


# --------------------------------------------------------------------------------------
# Defecto 1: el universo de clases salia de las verdades ENTREGADAS, que dependen del brazo.
# --------------------------------------------------------------------------------------


def test_el_universo_de_clases_no_puede_salir_de_lo_entregado() -> None:
    """The class universe is a property of the block, not of what a mechanism delivered.

    Es el defecto del denominador movil, que es la tesis del articulo, dentro del aparato del
    articulo. Aqui dos brazos entregan subconjuntos cuyas verdades difieren: si el universo se
    tomara de lo entregado, cada uno promediaria sobre clases distintas.
    """
    verdad_bloque = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    promesa = [0, 1, 2, 3]
    presentes = presentes_en_bloque(verdad_bloque)
    assert presentes == (0, 1, 2, 3)

    # Brazo A entrega parcelas de las clases 0 y 1; brazo B, de las clases 0, 1 y 2.
    a_verdad, a_pred = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
    b_verdad, b_pred = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 0, 1, 1, 2, 2])

    a = macro_over(a_verdad, a_pred, promesa, presentes=presentes)
    b = macro_over(b_verdad, b_pred, promesa, presentes=presentes)
    # Sobre el MISMO universo de cuatro clases, A cubre dos y B cubre tres, y se ve.
    assert a == pytest.approx(0.5)
    assert b == pytest.approx(0.75)

    # Y esto es lo que hacia la version anterior: tomar el universo de las verdades
    # ENTREGADAS. Los dos brazos salen perfectos y la diferencia entre ellos desaparece.
    a_defectuoso = macro_over(a_verdad, a_pred, promesa, presentes=presentes_en_bloque(a_verdad))
    b_defectuoso = macro_over(b_verdad, b_pred, promesa, presentes=presentes_en_bloque(b_verdad))
    assert a_defectuoso == b_defectuoso == pytest.approx(1.0)
    assert a_defectuoso > a and b_defectuoso > b, (
        "el denominador movil infla los dos brazos y borra la diferencia entre ellos"
    )


def test_presentes_es_obligatorio() -> None:
    """The parameter has no default, because a default is how this defect came back."""
    with pytest.raises(TypeError):
        macro_over(np.array([0, 1]), np.array([0, 1]), [0, 1])  # type: ignore[call-arg]


def test_una_clase_prometida_y_ausente_del_bloque_no_entra_como_cero() -> None:
    """A promised class absent from the block reports a failure that never happened."""
    verdad = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    presentes = presentes_en_bloque(verdad)
    assert macro_over(verdad, pred, [0, 1, 5], presentes=presentes) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Defecto 2: el umbral de confianza se elegia DENTRO del bloque que lo mide.
# --------------------------------------------------------------------------------------


def test_el_umbral_de_confianza_no_sale_del_bloque_evaluado(universo) -> None:
    """The operating point comes from the training blocks, so coverage is not forced to match.

    La implementacion anterior ordenaba las confianzas del bloque evaluado y cortaba en el
    recuento del mecanismo de referencia, con lo que la cobertura coincidia EXACTAMENTE en
    todos los bloques. Esa igualdad perfecta es la firma del defecto: igualar el recuento sin
    mirar dentro del bloque es imposible.
    """
    proba, labels, splits = universo
    libre = proba.argmax(axis=1)
    ref = frontier(
        proba,
        labels,
        splits,
        (3,),
        legend_fn=lambda train, k: legend_by_f1(labels, libre, train, k),
        mechanism="retirada por F1",
    )
    conf = confidence_baseline(proba, labels, splits, ref, num_classes=proba.shape[1])
    iguales = [
        int(a.delivered.sum()) == int(b.delivered.sum()) for a, b in zip(ref, conf, strict=True)
    ]
    assert not all(iguales), (
        "la cobertura coincide exactamente en todos los bloques: el umbral se esta "
        f"eligiendo dentro del bloque evaluado ({iguales})"
    )


# --------------------------------------------------------------------------------------
# Defecto 3: el intervalo se remuestreaba a nivel de parcela, no de bloque.
# --------------------------------------------------------------------------------------


def _puntos(deltas: list[float]) -> tuple[list[BlockPoint], list[BlockPoint]]:
    """Two aligned mechanisms whose per-block difference is exactly ``deltas``."""
    izq, der = [], []
    for i, d in enumerate(deltas):
        comun = {
            "mechanism": "x",
            "k": 3,
            "block": i,
            "legend": (0, 1, 2),
            "delivered": np.ones(4, dtype=bool),
            "emitted": np.zeros(4, dtype=int),
            "native_f1": 0.0,
            "accuracy": 0.0,
        }
        izq.append(BlockPoint(aligned_f1=0.5 + d, **comun))
        der.append(BlockPoint(aligned_f1=0.5, **comun))
    return izq, der


def test_la_unidad_de_remuestreo_es_obligatoria() -> None:
    """No default unit: the silent default is what hid the defect for two rounds."""
    izq, der = _puntos([0.01, 0.02, 0.03, 0.04])
    with pytest.raises(TypeError):
        paired_interval(np.zeros(4, dtype=int), [], izq, der)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="unidad de remuestreo"):
        paired_interval(np.zeros(4, dtype=int), [], izq, der, unidad="galaxia")


def test_el_intervalo_por_bloque_usa_los_bloques_como_unidad() -> None:
    """Four blocks give four degrees of freedom, whatever the parcel count is."""
    izq, der = _puntos([0.01, 0.02, 0.03, 0.04])
    r = paired_interval(np.zeros(4, dtype=int), [], izq, der, unidad="bloque")
    assert r["n_unidades"] == 4
    assert r["unidad"] == "bloque"
    assert r["ci_low"] < r["delta"] < r["ci_high"]
    # Comprobacion contra scipy: media 0,025, sd 0,0129, n=4.
    assert r["delta"] == pytest.approx(0.025)
    assert r["ci_low"] == pytest.approx(0.004458, abs=1e-5)
    assert r["ci_high"] == pytest.approx(0.045542, abs=1e-5)


def test_comparar_un_mecanismo_consigo_mismo_da_un_intervalo_degenerado() -> None:
    """The harness self-check: zero difference cannot produce a non-degenerate interval."""
    izq, der = _puntos([0.0, 0.0, 0.0, 0.0])
    r = paired_interval(np.zeros(4, dtype=int), [], izq, der, unidad="bloque")
    assert r["ci_low"] == r["ci_high"] == pytest.approx(0.0)
    assert r["p_valor"] == 1.0


def test_el_intervalo_por_bloque_es_mas_ancho_que_el_de_parcela(universo) -> None:
    """Resampling parcels inside blocks buys precision the design does not have.

    Es la razon de fondo del defecto: cinco bloques espaciales no son dieciseis mil replicas,
    y el intervalo de parcela responde a otra pregunta —cuanto se movería si las parcelas de
    ESTOS bloques se hubieran muestreado de otro modo— mucho mas estrecha.
    """
    proba, labels, splits = universo
    libre = proba.argmax(axis=1)
    izq = frontier(
        proba,
        labels,
        splits,
        (3,),
        legend_fn=lambda train, k: legend_by_f1(labels, libre, train, k),
        mechanism="retirada por F1",
    )
    der = confidence_baseline(proba, labels, splits, izq, num_classes=proba.shape[1])
    bloque = paired_interval(labels, splits, izq, der, unidad="bloque")
    parcela = paired_interval(
        labels, splits, izq, der, unidad="parcela", n_boot=200, random_state=0
    )
    ancho_bloque = bloque["ci_high"] - bloque["ci_low"]
    ancho_parcela = parcela["ci_high"] - parcela["ci_low"]
    assert ancho_bloque > ancho_parcela, (
        f"el intervalo por bloque ({ancho_bloque:.4f}) no es mas ancho que el de parcela "
        f"({ancho_parcela:.4f}): el remuestreo de parcelas esta comprando precision que el "
        "diseno no tiene"
    )


# --------------------------------------------------------------------------------------
# Ronda 4: la fuga que quedaba en el punto de operacion, y la inferencia donde no la hay.
# --------------------------------------------------------------------------------------


def test_el_umbral_no_depende_de_nada_del_bloque_de_prueba(universo) -> None:
    """Invariance, not a symptom: the frozen threshold cannot move with any test-side change.

    El test anterior comprobaba solo que los recuentos no coincidieran siempre, que es un
    sintoma. Una auditoria mostro la fuga que dejaba pasar: la TASA objetivo seguia siendo
    `ref.delivered.mean()`, o sea la cobertura que el otro mecanismo realizo EN EL BLOQUE DE
    PRUEBA. Manteniendo el entrenamiento fijo y cambiando solo la mascara de referencia, la
    entrega del comparador cambiaba. Aqui se comprueba el mecanismo: el umbral es funcion del
    entrenamiento y de la leyenda, y de nada mas.
    """
    proba, _, splits = universo
    train_pos, test_pos = splits[0]
    legend = (0, 1, 2)
    umbral = umbral_desde_entrenamiento(proba, train_pos, legend)

    # Se destroza el bloque de prueba entero: otras posteriores, otro orden, otro tamano.
    rng = np.random.default_rng(99)
    otro = proba.copy()
    otro[test_pos] = rng.dirichlet(np.ones(proba.shape[1]), size=test_pos.size)
    assert umbral_desde_entrenamiento(otro, train_pos, legend) == umbral

    # Y se comprueba que el umbral SI se mueve con el entrenamiento, para que el test no pase
    # por ser insensible a todo.
    otro_train = proba.copy()
    otro_train[train_pos] = rng.dirichlet(np.ones(proba.shape[1]) * 5, size=train_pos.size)
    assert umbral_desde_entrenamiento(otro_train, train_pos, legend) != umbral


def test_cambiar_la_referencia_no_cambia_lo_que_entrega_el_comparador(universo) -> None:
    """The comparator's delivery is a function of training and legend, not of the reference mask."""
    proba, labels, splits = universo
    libre = proba.argmax(axis=1)
    ref = frontier(
        proba,
        labels,
        splits,
        (3,),
        legend_fn=lambda train, k: legend_by_f1(labels, libre, train, k),
        mechanism="retirada por F1",
    )
    original = [b.delivered.copy() for b in confidence_baseline(proba, labels, splits, ref)]

    # Misma leyenda y mismo bloque, otra mascara de entrega en la referencia.
    manipulada = [
        BlockPoint(
            mechanism=r.mechanism,
            k=r.k,
            block=r.block,
            legend=r.legend,
            delivered=np.zeros_like(r.delivered),
            emitted=r.emitted,
            aligned_f1=r.aligned_f1,
            native_f1=r.native_f1,
            accuracy=r.accuracy,
        )
        for r in ref
    ]
    despues = [b.delivered for b in confidence_baseline(proba, labels, splits, manipulada)]
    for antes, ahora in zip(original, despues, strict=True):
        assert np.array_equal(antes, ahora), (
            "la entrega del comparador cambia al tocar la mascara de la referencia: el punto de "
            "operacion sigue leyendo el bloque de prueba"
        )


def test_con_dos_bloques_no_hay_intervalo_ni_p() -> None:
    """Two regions are two numbers. The contract says report both deltas and nothing else.

    El productor de BreizhCrops tiene exactamente dos regiones y le estaba aplicando Holm a un
    p que no deberia existir.
    """
    izq, der = _puntos([0.01, 0.03])
    r = paired_interval(np.zeros(4, dtype=int), [], izq, der, unidad="bloque")
    assert r["ci_low"] is None
    assert r["ci_high"] is None
    assert r["p_valor"] is None
    assert r["excluye_cero"] is None
    assert r["deltas_por_bloque"] == pytest.approx([0.01, 0.03])
    assert "no se publica intervalo ni p" in r["motivo"]


def test_varianza_cero_con_efecto_no_nulo_no_se_confunde_con_no_efecto() -> None:
    """A constant non-zero difference is not "no difference", and there is no t to compute.

    La rama de varianza cero devolvia p=1 y «no excluye el cero» para CUALQUIER constante, y el
    unico test de esa rama usaba cuatro ceros: otra vez el unico valor que no distingue el bug.
    """
    izq, der = _puntos([0.1, 0.1, 0.1, 0.1])
    r = paired_interval(np.zeros(4, dtype=int), [], izq, der, unidad="bloque")
    assert r["ci_low"] == r["ci_high"] == pytest.approx(0.1)
    assert r["excluye_cero"] == pytest.approx(1.0), "un intervalo en 0,1 no contiene el cero"
    assert r["p_valor"] is None, "sin varianza no hay t, y no se inventa un p"
    assert "no esta definida" in r["motivo"]


def test_lo_indefinido_es_nan_y_no_un_cero_con_pinta_de_medida() -> None:
    """An undefined cell must be visible to whoever averages it, not a perfect-failure zero."""
    vacio = np.array([], dtype=int)
    assert np.isnan(macro_over(vacio, vacio, [0, 1], presentes=(0, 1)))
    verdad = np.array([0, 0, 1])
    assert np.isnan(macro_over(verdad, verdad, [7, 8], presentes=presentes_en_bloque(verdad)))


def test_un_bloque_indefinido_se_cuenta_en_vez_de_promediarse() -> None:
    """The block interval drops undefined blocks and says how many it dropped."""
    izq, der = _puntos([0.02, 0.04, 0.06, float("nan")])
    r = paired_interval(np.zeros(4, dtype=int), [], izq, der, unidad="bloque")
    assert r["bloques_indefinidos"] == 1
    assert r["n_unidades"] == 3
    assert r["p_valor"] is not None
