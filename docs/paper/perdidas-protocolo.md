# Protocolo de elicitación de la tabla de pérdidas (US-172)

**Estado**: BORRADOR para congelar. Se congela con un commit fechado **antes de contactar a nadie**,
y a partir de ahí toda desviación entra como enmienda fechada. **Ninguna respuesta se recoge antes
de resolver §4.**

Este documento es el paso 1 del camino crítico. No produce ninguna cifra y no mira ningún resultado:
por eso puede escribirse ahora, y por eso escribirlo no compromete nada.

---

## 1. Qué se elicita, y qué NO

Se elicita **el precio de cada resultado posible del mapa, para cada afectado**. Nada más.

Un mapa de cultivos que no alcanza calidad puede fallar de cuatro maneras cualitativamente
distintas, y el artículo entero depende de que no cuesten lo mismo:

| Resultado | Qué recibe quien usa el mapa |
|---|---|
| **Etiqueta errónea** | Una clase concreta y equivocada, sin aviso |
| **No respuesta** | La parcela queda sin clasificar |
| **Conjunto ambiguo** | Varias clases plausibles, sin decidir entre ellas |
| **Retroceso taxonómico** | Una clase más gruesa: cierta, pero menos útil |

**Lo que este protocolo NO hace, y hay que decirlo porque es la tentación**: no pregunta qué
mecanismo prefieren, no enseña resultados nuestros, no valida una tabla que ya tengamos, y no pide a
nadie que rellene una matriz. Se pregunta por **consecuencias en su trabajo**, y los números salen
de ahí.

## 2. Los dos afectados, y por qué son dos

| Grupo | Quién | Por qué no pierde lo mismo |
|---|---|---|
| **A · Control y verificación** | Quien usa el mapa para verificar declaraciones de superficie o asignar controles sobre el terreno | Una etiqueta errónea puede producir una decisión administrativa injusta; un silencio solo cuesta una visita más |
| **B · Asesoría al productor** | Quien usa el mapa para aconsejar en campo o planificar insumos | Un silencio deja al productor sin recomendación en su ventana agronómica; una etiqueta errónea se detecta al pisar la parcela |

**Esa asimetría es la hipótesis del artículo**, y por eso hay que medirla, no suponerla. Si los dos
grupos dieran la misma tabla, el resultado sería negativo y se publicaría igual.

**Mínimo: tres informantes por grupo.** No es una muestra estadística y no se presenta como tal: es
el mínimo para que un desacuerdo dentro del grupo sea visible.

## 3. Quién NO puede ser informante

- Nadie del equipo del artículo, ni sus coautores.
- Nadie que haya visto resultados nuestros sobre estos mecanismos.
- Nadie cuya relación con el equipo haga costoso decir que no.

**El motivo es el criterio de aceptación**: un precio que solo hemos dicho nosotros no es una
elicitación, es un supuesto con testigos.

## 4. Antes de contactar a nadie

Cuatro cosas resueltas y escritas, **en este orden**, y hasta que las cuatro estén, no se envía ni
un mensaje:

1. **¿Exige revisión ética la institución para esto?** Se pregunta explícitamente, se guarda la
   respuesta por escrito, y **si la exige, no se recoge ninguna respuesta antes de la aprobación**.
   Un «probablemente no haga falta» no cuenta como respuesta.
2. **Consentimiento informado**: qué se pregunta, para qué, cuánto dura, que es voluntario, que se
   puede parar en cualquier momento y retirar lo dicho, y que lo publicado será anónimo. Anexo A.
3. **Minimización**: no se pide nombre, empleador identificable, ni ningún dato personal que no sea
   el grupo (A o B) y los años de experiencia en tramos.
4. **Acceso, retención y anonimización**: quién puede leer las respuestas crudas, dónde viven,
   cuánto tiempo, y con qué identificador se publican (`A1`…`A3`, `B1`…`B3`).

## 5. El instrumento

Una conversación de unos 40 minutos, con guion. Se graba solo si la persona lo autoriza por
separado; si no, se toman notas y se le leen al final para que las corrija.

### 5.1 Contexto (sin números nuestros)

> «Trabajo en un sistema que dice qué se cultiva en cada parcela a partir de imágenes de satélite.
> Cuando el sistema no está seguro, puede hacer cuatro cosas distintas, y quiero entender qué le
> cuesta a usted cada una. No hay respuestas correctas y no le voy a enseñar ningún resultado.»

### 5.2 Primero el orden, después la magnitud

**Se pregunta el orden antes que cualquier número.** Pedir una cifra en frío ancla, y el ancla la
pondríamos nosotros.

> «Para una parcela de la que depende una decisión suya: ordene estas cuatro situaciones de la que
> menos le cuesta a la que más. Si dos empatan, dígalo.»

Se leen las cuatro **como situaciones concretas de su trabajo**, no como categorías:

- «El mapa dice trigo y era cebada, y usted actúa con eso.»
- «El mapa no dice nada de esa parcela.»
- «El mapa dice: es trigo o es cebada, decida usted.»
- «El mapa dice cereal de invierno, sin precisar cuál.»

Después, y solo después, la magnitud, con un **intercambio concreto**:

> «Tomando como unidad la situación que ha puesto en el medio: ¿cuántas parcelas de esa estaría
> dispuesto a aceptar para evitar una de la peor?»

De ahí sale una razón, no un valor absoluto — que es exactamente lo que la tabla necesita.

### 5.3 Las preguntas que buscan el desacuerdo

> - «¿Cambia su orden si la parcela es de una clase rara?»
> - «¿Cambia si es época de decidir, frente a un análisis a posteriori?»
> - «¿Hay alguna de las cuatro que no debería ocurrir nunca, cueste lo que cueste?» *(busca un
>   coste no compensatorio, que rompería la escala aditiva y hay que saberlo)*
> - «¿Qué le he preguntado mal?»

## 6. Custodia de las respuestas

| | |
|---|---|
| Crudas | `data/perdidas/respuestas/<grupo><n>.md`, versionadas con DVC, **nunca en git** |
| Sello | MD5 de cada fichero en `paper/ARTIFACTS.md`, como cualquier artefacto |
| Publicable | `docs/paper/perdidas.md`, con la tabla y la trazabilidad celda → respuestas |
| Identificadores | `A1`…`A3`, `B1`…`B3`. Ningún nombre, ningún empleador |

## 7. De las respuestas a la tabla

1. Cada informante produce **un orden** y **una o más razones** de intercambio.
2. La celda `L(resultado, acción, afectado)` se sintetiza por grupo, y **cada celda cita los
   identificadores que la sostienen**. Una celda sin cita no entra.
3. **El desacuerdo se publica, no se promedia.** Si dentro de un grupo hay dos órdenes distintos, la
   tabla publica el rango y el artículo declara qué hace con él: normalmente, correr el análisis en
   los dos extremos y reportar los dos.
4. Si aparece un coste **no compensatorio** —«eso no debe ocurrir nunca»—, la escala aditiva no
   vale para ese afectado y se dice, en vez de forzarlo a un número.

## 8. Qué invalidaría esta elicitación

Se declara antes para no discutirlo después:

- Que un informante haya visto resultados nuestros.
- Que la magnitud se pregunte antes que el orden.
- Que una celda no pueda citar ninguna respuesta.
- Que el desacuerdo se resuelva promediando.
- Que se recoja una sola respuesta antes de cerrar §4.

## 9. Congelación

Este protocolo se congela con un commit fechado antes del primer contacto. Toda desviación entra
como **enmienda fechada antes** de recoger la respuesta afectada, con su motivo. La tabla resultante
se declara en el preregistro **antes de calcular nada**, y de ella —y solo de ella— salen el margen
práctico de US-174 y la geometría de US-175.

---

## Anexo A · Consentimiento (texto para leer o enviar)

> Le pido unos 40 minutos para entender qué le cuesta, en su trabajo, cada una de las cuatro
> maneras en que un mapa de cultivos automático puede fallar. Es para un artículo académico sobre
> cómo evaluar estos sistemas.
>
> **Qué se guarda**: sus respuestas, sin su nombre ni el de su organización. Se le identifica como
> «informante A2», por ejemplo. Se guardan las respuestas y no la grabación, salvo que usted
> autorice la grabación por separado.
>
> **Qué NO se le pide**: ningún dato personal más allá del tipo de trabajo y sus años de
> experiencia en tramos.
>
> **Sus derechos**: participar es voluntario, puede parar en cualquier momento sin dar explicación,
> y puede pedir que se retire lo que ha dicho hasta la publicación del artículo.
>
> **Qué se publica**: una tabla de costes relativos y, si hay desacuerdo entre informantes, el
> rango. Sus frases pueden citarse anonimizadas; si prefiere que no, dígalo y no se citan.
>
> ¿Quiere que sigamos?

## Anexo B · Lo que este protocolo no puede resolver, y hay que decir en el artículo

- Seis informantes no son una muestra representativa, y la tabla no se presenta como tal: es una
  **declaración de costes con procedencia**, que es infinitamente más de lo que tenía el artículo
  cuando la cardinalidad ocupaba ese lugar.
- Dos grupos no agotan los afectados. Faltan al menos el productor y la administración estadística,
  y su ausencia se declara como límite de alcance.
- Un coste declarado en entrevista no es un coste revelado en la práctica. Se dice.
