# Auditoría externa

## Por qué

El plan lleva ocho auditorías internas encima —cobertura, ejecutabilidad, estratégica, red team,
forense, cumplimiento, novedad y estadística— y todas fueron útiles. Pero todas las corrió el mismo
equipo, con las mismas herramientas y sobre el mismo contexto, así que comparten puntos ciegos por
construcción. Varias veces el hallazgo que más costó vino de mirar algo que el equipo trataba como
un dato y resultó ser una elección suya.

## Cómo se usa

1. Se entrega [`prompt-auditoria-externa.md`](prompt-auditoria-externa.md) a alguien de fuera, con
   acceso de lectura a los dos repositorios.
2. **La parte A se responde antes de leer nada nuestro.** Ese orden es el encargo, no una
   formalidad: si lee primero nuestras conclusiones, su hallazgo deja de ser independiente y pasa a
   ser una confirmación, que vale mucho menos.
3. La parte B contrasta, y su pregunta más valiosa es la 8: qué encontró que no está en nuestra
   lista.

## Qué se le da y qué no

**Se le da**: los dos repositorios, los artefactos, el registro de custodia con sus MD5, y las dos
comprobaciones mecánicas (`make plan-check`, `make paper-artifacts-check`).

**No se le da por adelantado**: nuestras conclusiones, la lista de defectos ya corregidos, ni el
histórico de por qué se retiró el intento anterior. Todo eso está en el repositorio y lo leerá en la
parte B, cuando ya tenga su propio juicio formado.

## Qué esperar

Las auditorías anteriores han encontrado, entre otras cosas: un estimando que cometía el error que
el propio artículo denunciaba, un intervalo construido sobre la unidad de remuestreo equivocada, una
premisa atribuida al equipo que sus propios artefactos desmienten, un gate que no podía detectar
tres de sus dieciséis tokens, y un parámetro tratado como restricción del dato que era el valor por
defecto de una función.

No esperamos que la externa encuentre menos.
