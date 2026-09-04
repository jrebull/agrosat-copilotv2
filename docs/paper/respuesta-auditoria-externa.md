# Respuesta a la auditoría externa · ronda 1

**Regla de esta página**: un hallazgo no se cierra con una historia de usuario. Se cierra con un cambio de comportamiento y con dónde comprobarlo. Lo que siga siendo futuro se marca **ABIERTO** y se dice por qué.

| # | Hallazgo | Estado | Dónde comprobarlo |
|---|---|---|---|
| 1 | La utilidad no podía cobrar la abstención | **CERRADO** | `ml/eval/set_valued.py`: el valor de abstenerse es un término aparte y sin valor por defecto. Dos tests de regresión en `tests/ml/eval/test_set_valued.py` que fallan con la implementación anterior |
| 1b | La cardinalidad no es el coste | **ABIERTO, con dueño** | US-172 lo sustituye por una tabla de pérdidas por acción, resultado y afectado. No está hecho |
| 2 | Tres razones libres, no dos | **ABIERTO, con dueño** | US-175. La afirmación errónea está retirada del plan y del preregistro |
| 3 | Más bloques no son más réplicas | **PARCIAL** | El barrido de `k` deja de presentarse como palanca de potencia, en US-171 y en el preregistro. Falta declarar el estimando: US-173 |
| 4 | Escoger el resultado después de verlo | **PARCIAL** | Retirada del plan la regla de mover la primaria «donde haya potencia». La banda de equivalencia deja de anclarse en el MDE. Falta fijar el margen práctico: US-174 |
| 5 | Multiplicidad sobre la superficie entera | **ABIERTO, con dueño** | US-128 y US-139 |
| 6a | Cobertura del despliegue de otro modelo | **CERRADO** | 0,88269 —14 688 de 16 640— en el plan, el cuaderno y el reencuadre. Verificado contra `cardinalidad.json` y el soporte sellado |
| 6b | Ledger: 12 páginas de un PDF de 15, 43 entradas de un fichero con 44 | **CERRADO** | `paper/ARTIFACTS.md` |
| 6c | Cabecera del plan que no coincidía con su parser | **CERRADO** | `plan.html`, y `make plan-check` da los números |
| 6d | Presupuesto de páginas que sumaba 20,5 | **CERRADO** | `docs/paper/campo-de-tiro.md`, con el total explícito |
| 6e | MDE aproximado en vez de t no central | **ABIERTO** | Reconocido. Las cifras del plan son la aproximación normal; hay que marcarlas como tales o recomputarlas |
| 9 | «Reparado en US-124/125» cuando seguía sin repararse | **CERRADO** | `campo-de-tiro.md` dice ahora AÚN NO REPARADO en las dos filas |
| 10 | Errores atribuidos a causa local con causa más profunda | **PARCIAL** | El preregistro recoge la distinción entre no detectado y equivalente. Falta el estimando: US-173 |

## Lo que cambió en el plan

La causa raíz que señalaste —que el problema de decisión no tiene función de pérdida identificada— pasa a ser una épica propia, la **27**, colocada delante de todas las demás. El camino crítico ahora empieza en `US-172`, la tabla de pérdidas, y no en el protocolo.

## Lo que NO hemos hecho y sabemos

- No hemos recomputado el MDE con la t no central.
- No hemos tocado la selección de predictores, que sigue usando las etiquetas de evaluación.
- El preregistro volvió a estado de borrador con tres parámetros abiertos, y **no se firma hasta cerrarlos**.
