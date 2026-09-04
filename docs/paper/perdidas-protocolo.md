# Protocolo de elicitación de la tabla de pérdidas (US-172)

**Versión 0.2**, tras la revisión que encontró una **falla de identificación anterior a cualquier
entrevista**: la versión 0.1 elicitaba un orden y **una sola razón** para cuatro pérdidas, y cuatro
pérdidas tienen tres grados de libertad una vez fijada la escala. Dos magnitudes quedaban sin
identificar y las habríamos rellenado nosotros, que es exactamente la salida por «supuesto del
artículo» que US-172 prohíbe, entrando por la puerta de atrás.

**Estado**: BORRADOR, con los campos operativos **sin rellenar** (§10 bis). Se congela con un commit fechado **antes de reclutar a nadie** y **después** de la determinación institucional, y desde ahí
toda desviación entra como enmienda fechada. Este documento no produce ninguna cifra y no mira
ningún resultado: por eso puede escribirse ahora.

---

## 1. Qué se elicita, y qué NO

Se elicita **el precio de cada par (acción del sistema, resultado real)** para cada afectado. Nada
más.

**Lo que este protocolo NO hace**: no pregunta qué mecanismo prefieren, no enseña resultados
nuestros, no valida una tabla que ya tengamos, y no pide a nadie que rellene una matriz. Se pregunta
por consecuencias en su trabajo, y los números salen de ahí.

## 2. El espacio de resultados, entero y declarado antes

La versión 0.1 prometía «cada resultado posible» y presentaba cuatro casos escogidos. **Las celdas
que faltaban habrían aparecido después, y habrían sido supuestos nuestros.** Aquí está la rejilla
completa, y es la que puntuará el código:

| id | Acción del sistema | Resultado real | Qué recibe quien usa el mapa |
|---|---|---|---|
| `R0` | Etiqueta precisa | **acierta** | La clase correcta. **Referencia: pérdida 0 por definición** |
| `L1` | Etiqueta precisa | **falla** | Una clase concreta y equivocada, sin aviso |
| `L2` | No responde | — | La parcela queda sin clasificar |
| `L3` | Conjunto de clases | **contiene la verdad** | Varias plausibles, una de ellas es la buena |
| `L4` | Conjunto de clases | **no contiene la verdad** | Varias plausibles y ninguna es la buena |
| `L5` | Clase gruesa | **la verdad cae dentro** | Cierta, pero menos útil |
| `L6` | Clase gruesa | **la verdad cae fuera** | Menos útil **y** equivocada |

**Siete celdas, no cuatro.** `L4` y `L6` faltaban, y son justo las que hacen que un conjunto o un
retroceso no sean gratis. `R0` faltaba como referencia explícita, y sin referencia no hay escala.

**Una pregunta más, y es la del artículo**: ¿cambia `L3` si el conjunto son dos clases o cinco? Se
pregunta, no se supone. Si la respuesta es que no cambia, la cardinalidad no entra en la pérdida ni
siquiera como argumento, y eso es un resultado.

## 3. Identificación: cuántos intercambios hacen falta

Con `R0 = 0` y **una** pérdida fijada como unidad de escala, quedan **cinco magnitudes libres**. Un
orden y una razón dan dos números: faltaban tres.

> **Regla**: se eliciten **al menos cinco intercambios independientes** más **dos comprobaciones de
> consistencia**. Si no se completan, la tabla de ese informante queda **incompleta y se declara
> así**; no se rellena por interpolación.

**Cómo se consigue en una entrevista y no en un examen:**

1. **Orden completo** de las seis pérdidas (`L1`…`L6`). Es barato, es robusto y no ancla.
2. **Cadena de intercambios adyacentes**: entre cada par consecutivo del orden. Cinco pares, cinco
   razones, y cada una es una pregunta concreta y fácil.
3. **Dos comprobaciones**: un intercambio **directo** entre la primera y la última del orden, y otro
   **saltando una posición**. La cadena predice esos dos valores; si no coinciden dentro del margen
   declarado, **se registra la inconsistencia y se reporta**, no se corrige.

**Escala**: la unidad es `L1` —la etiqueta precisa equivocada— porque es la que todo el mundo
entiende sin explicación. Se dice en voz alta que es solo una unidad de medida y no un juicio sobre
cuál importa más.

**Margen de consistencia**: se declara aquí, antes de oír a nadie. Un factor de **2** entre el valor
encadenado y el directo se considera compatible con la imprecisión de la tarea; por encima, el
informante queda marcado como **internamente inconsistente** y su vector se reporta con esa
etiqueta. No se descarta: una inconsistencia es un dato sobre lo difícil que es la pregunta.

## 4. Los dos afectados, y por qué son dos

| Grupo | Quién | Por qué no pierde lo mismo |
|---|---|---|
| **A · Control y verificación** | Quien usa el mapa para verificar declaraciones de superficie o asignar controles sobre el terreno | Una etiqueta errónea puede producir una decisión administrativa injusta; un silencio solo cuesta una visita más |
| **B · Asesoría al productor** | Quien usa el mapa para aconsejar en campo o planificar insumos | Un silencio deja al productor sin recomendación en su ventana agronómica; una etiqueta errónea se detecta al pisar la parcela |

**Esa asimetría es la hipótesis del artículo**, y por eso hay que medirla en vez de suponerla. Si los
dos grupos dieran la misma tabla, el resultado sería negativo y se publicaría igual.

**Mínimo tres informantes por grupo.** No es una muestra estadística y no se presenta como tal: es el
mínimo para que un desacuerdo dentro del grupo sea visible.

## 5. Quién NO puede ser informante

- Nadie del equipo del artículo, ni sus coautores.
- Nadie que haya visto resultados nuestros sobre estos mecanismos.
- Nadie cuya relación con el equipo haga costoso decir que no.

**El motivo es el criterio de aceptación**: un precio que solo hemos dicho nosotros no es una
elicitación, es un supuesto con testigos.

## 6. Antes de reclutar a nadie

**Se puede —y hay que— enviar la consulta institucional del punto 1.** La versión 0.1 decía «no se
envía ni un mensaje», que bloqueaba el propio mensaje necesario para desbloquear. Lo que no se hace
antes de cerrar los cuatro puntos es **contactar o reclutar a un informante**.

1. **¿Exige revisión ética la institución para esto?** Se pregunta por escrito (anexo C), se guarda
   la respuesta, y **si la exige, no se recoge ninguna respuesta antes de la aprobación**. Un
   «probablemente no haga falta» no cuenta como respuesta.
2. **Consentimiento informado** (anexo A).
3. **Minimización**: no se pide nombre, empleador identificable, ni ningún dato personal más allá
   del grupo (A o B) y los años de experiencia en tramos.
4. **Custodia**, y con nombres propios — **DVC no es control de acceso**, solo versionado:

| | |
|---|---|
| **Custodio** | Una persona nombrada del equipo, escrita aquí antes de empezar |
| **Dónde** | Almacenamiento cifrado con acceso restringido al custodio, **fuera del repositorio**. En el repositorio solo entra el material ya anonimizado |
| **Permisos** | Lista nominal de quién puede leer las respuestas crudas; cualquier ampliación se anota con fecha y motivo |
| **Vínculo reversible** | Una tabla `identificador ↔ contacto` que vive **solo** con el custodio, para poder localizar una respuesta si alguien pide retirarla |
| **Retención** | Hasta la publicación del artículo o 24 meses, lo que ocurra antes |
| **Destrucción** | La tabla de vínculo se destruye al cumplirse el plazo; queda constancia de la destrucción |
| **Retirada** | A petición del informante, en cualquier momento hasta la publicación: se localiza por el vínculo, se borra la respuesta cruda y se recalcula la tabla sin ella |

## 7. El instrumento

Entre 45 y 60 minutos, con guion. Se puede partir en dos sesiones. Se graba **solo** con
autorización separada; si no, se toman notas y se le leen al final para que las corrija.

### 7.1 Contexto, sin números nuestros

> «Trabajo en un sistema que dice qué se cultiva en cada parcela a partir de imágenes de satélite.
> Cuando el sistema no está seguro puede hacer varias cosas distintas, y quiero entender qué le
> cuesta a usted cada una. No hay respuestas correctas y no le voy a enseñar ningún resultado.»

### 7.2 Las seis situaciones, en orden aleatorizado

Se leen **como situaciones concretas de su trabajo**, nunca como categorías:

- `L1` «El mapa dice trigo, era cebada, y usted actúa con eso.»
- `L2` «El mapa no dice nada de esa parcela.»
- `L3` «El mapa dice: es trigo o es cebada, decida usted — y era una de las dos.»
- `L4` «El mapa dice: es trigo o es cebada, decida usted — y no era ninguna.»
- `L5` «El mapa dice cereal de invierno, sin precisar cuál — y lo era.»
- `L6` «El mapa dice cereal de invierno — y era un cultivo de primavera.»

> **El orden de lectura se BLOQUEA por grupo, no se aleatoriza por informante.** Antes de la primera
> entrevista se generan tres permutaciones `P1`–`P3`. **Cada grupo recibe exactamente una vez cada
> permutación**; dentro del grupo se asignan al azar. Así los dos grupos tienen la misma
> distribución de órdenes de lectura. Si se incorporan más informantes, las asignaciones continúan
> balanceadas por grupo.
>
> Aleatorizar cada entrevista por separado —que es lo que decía la versión anterior— **todavía
> permite que grupo y orden queden confundidos por azar**: con tres informantes por grupo, no es
> improbable que un grupo reciba órdenes sistemáticamente distintos del otro. Y el efecto de grupo
> es exactamente el efecto que el artículo quiere medir.

**Las tres permutaciones, generadas y registradas antes de entrevistar** (semilla `20260904`,
`random.Random`, sin reemplazo entre permutaciones):

| | Orden de lectura |
|---|---|
| `P1` | `L3` · `L6` · `L5` · `L4` · `L2` · `L1` |
| `P2` | `L3` · `L5` · `L1` · `L6` · `L4` · `L2` |
| `P3` | `L1` · `L2` · `L5` · `L4` · `L3` · `L6` |

La asignación de cada permutación a cada informante se anota en su ficha **antes** de la entrevista.

### 7.3 Primero el orden, después la magnitud

**El orden se pregunta antes que cualquier número.** Pedir una cifra en frío ancla, y el ancla la
pondríamos nosotros.

> «Ordene estas seis situaciones, de la que menos le cuesta a la que más. Si dos empatan, dígalo.»

Después, la cadena de intercambios, con `L1` como unidad:

> «Piense en la situación que ha puesto justo debajo de esta otra. ¿Cuántas parcelas de la primera
> aceptaría para evitar una de la segunda?»

Y las dos comprobaciones, al final y sin avisar de que lo son:

> «Y ahora, directamente: ¿cuántas de la que puso primera aceptaría para evitar una de la última?»

### 7.4 Las preguntas que buscan el desacuerdo y las rupturas

> - «¿Cambia algo si el conjunto son cinco clases en vez de dos?» *(¿entra la cardinalidad?)*
> - «¿Cambia su orden si la parcela es de una clase rara?»
> - «¿Cambia si es época de decidir, frente a un análisis a posteriori?»
> - «¿Hay alguna de las seis que **no deba ocurrir nunca**, cueste lo que cueste?»
> - «¿Qué le he preguntado mal?»

## 8. Qué se hace con lo que salga: las ramas, declaradas antes

### 8.1 Desacuerdo entre informantes

**Se conserva el vector completo de cada informante y se analiza sobre esos vectores.** No se
construye una tabla combinando el mínimo de una celda con el máximo de otra: eso fabrica una tabla
que **ningún informante sostuvo** y le pone nuestra firma.

> **Regla preregistrada**: el análisis principal se corre **una vez por vector de informante**, y se
> reporta la dispersión del resultado entre vectores. Si se necesita un resumen por grupo, es la
> **mediana por celda del grupo declarada como tal**, y se dice que ese vector resumen puede no
> corresponder a ninguna persona. Cualquier envolvente —peor caso, mejor caso— se declara aquí antes
> de verla, y tiene que ser un vector **coherente**, no una mezcla celda a celda.

### 8.2 Coste no compensatorio

Detectar la prohibición no basta; hay que decir antes qué se hace con ella:

1. **Confirmar que es una prohibición y no un «muy caro»**: se ofrece un intercambio extremo. Si con
   un número suficientemente grande acepta, es compensatorio y entra en la escala.
2. Si no acepta a ningún precio, **se impone una restricción lexicográfica** para ese afectado: los
   mecanismos que producen esa celda con probabilidad no nula quedan **descartados antes de comparar
   nada**, y se dice cuáles.
3. **La suma deja de usarse para ese afectado.** Su resultado se reporta aparte, con la restricción
   escrita, y no se promedia con los demás.

### 8.3 Inconsistencia interna

Vector marcado, reportado con la etiqueta, y **no descartado**. Se declara cuántos informantes
quedaron marcados: si son mayoría, el problema es la tarea y no las personas, y eso también es un
resultado publicable.

## 9. Custodia de los datos

| | |
|---|---|
| Crudas, con vínculo | Solo con el custodio, cifradas, fuera del repositorio |
| Anonimizadas | `data/perdidas/respuestas/<grupo><n>.md`, versionadas con DVC, **nunca en git** |
| Sello | MD5 de cada fichero en `paper/ARTIFACTS.md`, como cualquier artefacto |
| Publicable | `docs/paper/perdidas.md`, con la tabla y la trazabilidad celda → identificadores |
| Ficha por informante | Grupo, tramo de experiencia, **permutación de lectura usada**, fecha |

## 10. Qué invalidaría esta elicitación

Declarado antes para no discutirlo después:

- Que un informante haya visto resultados nuestros.
- Que la magnitud se pregunte antes que el orden.
- Que una celda de la rejilla no pueda citar ninguna respuesta.
- Que el desacuerdo se resuelva mezclando celdas de informantes distintos.
- Que se lea la misma permutación a todos.
- Que se recoja una sola respuesta antes de cerrar §6.

## 10 bis. Campos operativos · **SIN RELLENAR**

Estos campos son responsabilidad de personas, no del repositorio, y **el protocolo no puede pasar a
`CONGELADO` mientras alguno diga `[POR DEFINIR]`**. Lo comprueba `make protocolo-check`.

| Campo | Valor |
|---|---|
| Investigador responsable | `[POR DEFINIR]` |
| Profesor responsable, si el anterior es estudiante | `[POR DEFINIR]` |
| Custodio de las respuestas crudas | `[POR DEFINIR]` |
| Plataforma cifrada concreta | `[POR DEFINIR]` |
| Personas nominalmente autorizadas a leer lo crudo | `[POR DEFINIR]` |
| Plazo de destrucción de las grabaciones | `[POR DEFINIR]` |
| Cómo pide una persona retirar sus datos | `[POR DEFINIR]` |
| Quién transcribe y quién valida la transcripción | `[POR DEFINIR]` |
| Referencia de la determinación o aprobación institucional | `[POR DEFINIR]` |

**Aquí no se escriben secretos, rutas privadas ni contactos personales.** Un nombre y un rol bastan;
el correo de cada persona vive en el fichero de vínculo del custodio, fuera del repositorio.

## 11. Congelación

Se congela con un commit fechado antes del primer reclutamiento. Toda desviación entra como
**enmienda fechada antes** de recoger la respuesta afectada. La tabla resultante se declara en el
preregistro **antes de calcular nada**, y de ella —y solo de ella— salen el margen práctico de
US-174 y la geometría de US-175.

---

## Anexo A · Consentimiento

> Le pido entre 45 y 60 minutos para entender qué le cuesta, en su trabajo, cada una de las maneras
> en que un mapa de cultivos automático puede fallar. Es para un artículo académico sobre cómo
> evaluar estos sistemas.
>
> **Qué se guarda**: sus respuestas, sin su nombre ni el de su organización. Se le identifica como
> «informante A2», por ejemplo. Se guardan las respuestas y no la grabación, salvo que usted
> autorice la grabación por separado.
>
> **Qué NO se le pide**: ningún dato personal más allá del tipo de trabajo y sus años de experiencia
> en tramos.
>
> **Quién puede leerlo**: una sola persona del equipo custodia las respuestas con su vínculo a
> usted, en almacenamiento cifrado. El resto del equipo solo ve el material anonimizado.
>
> **Sus derechos**: participar es voluntario, puede parar en cualquier momento sin dar explicación, y
> puede pedir que se retire lo que ha dicho en cualquier momento hasta la publicación. Si lo pide, se
> localiza su respuesta, se borra y se rehacen los cálculos sin ella.
>
> **Cuánto tiempo**: el vínculo entre usted y su identificador se destruye al publicarse el artículo
> o a los 24 meses, lo que ocurra antes.
>
> **Qué se publica**: una tabla de costes relativos y, si hay desacuerdo entre informantes, los
> vectores por separado. Sus frases pueden citarse anonimizadas; si prefiere que no, dígalo y no se
> citan.
>
> ¿Quiere que sigamos?

## Anexo B · Lo que este protocolo no puede resolver, y va en el artículo

- Seis informantes no son una muestra representativa, y la tabla no se presenta como tal: es una
  **declaración de costes con procedencia**, que es más de lo que había cuando la cardinalidad
  ocupaba ese lugar.
- Dos grupos no agotan los afectados. Faltan al menos el productor y la administración estadística,
  y su ausencia se declara como límite de alcance.
- Un coste declarado en entrevista no es un coste revelado en la práctica. Se dice.
- La rejilla de siete celdas trata cada par (acción, resultado) como un precio único. Si el precio
  depende de la clase concreta —y la pregunta de §7.4 sobre clases raras existe para detectarlo—, el
  modelo es una simplificación y se declara.

## Anexo C · Consulta institucional (texto para enviar antes de reclutar)

> Asunto: consulta sobre requisito de revisión ética para entrevistas metodológicas
>
> Preparo un artículo académico sobre evaluación de sistemas de clasificación de cultivos. Necesito
> entrevistar a entre seis y ocho profesionales para conocer **el coste relativo que tienen para su
> trabajo distintos tipos de error de un mapa automático**. Las entrevistas duran entre 45 y 60
> minutos.
>
> Para gestionar la invitación y el derecho de retirada, el custodio conservará nombre y medio de
> contacto en un fichero de vínculo separado, cifrado y con acceso restringido. El conjunto
> analítico solo contendrá grupo y experiencia por tramos. Si la persona autoriza una grabación, su
> voz se tratará como dato personal y la grabación se destruirá después de validar la
> transcripción.
>
> No se pregunta por personas identificables, ni por datos de sus organizaciones, ni por ningún dato
> sensible. Las respuestas se publican de forma anonimizada. Hay consentimiento informado por
> escrito, participación voluntaria y derecho de retirada hasta la publicación.
>
> **Mi pregunta es concreta**: ¿requiere este estudio revisión o aprobación de un comité de ética de
> la institución? Y si la requiere, ¿cuál es el procedimiento y el plazo estimado?
>
> No contactaré a ningún participante hasta tener su respuesta por escrito.

## Anexo D · Filtro de elegibilidad (para el primer contacto)

Tres preguntas, antes de agendar nada. Cualquier «sí» en las dos últimas excluye.

1. «En su trabajo, ¿usa mapas de cultivos —propios o de terceros— para **verificar declaraciones o
   asignar controles** (grupo A), o para **aconsejar a productores o planificar insumos** (grupo B)?»
   Si no es ninguno de los dos, no es informante de este estudio.
2. «¿Ha visto resultados de nuestro sistema o de este artículo?»
3. «¿Tiene alguna relación laboral, contractual o académica con el equipo?»

Se registra el resultado del filtro de **todos** los contactados, incluidos los excluidos y los que
declinan, para poder decir en el artículo a cuántos se preguntó.
