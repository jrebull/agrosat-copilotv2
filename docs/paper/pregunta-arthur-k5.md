# La pregunta para Arthur, antes de firmar el preregistro

**Contexto**: una vez firmado el preregistro no se toca, y fija el parámetro que decide si hay resultado. Por eso esto va antes.

---

## El mensaje, para enviar tal cual

> Arthur, una decisión que necesito que veas antes de firmar el preregistro, porque después no se puede cambiar y nos cuesta el resultado principal.
>
> **El número de bloques espaciales con que evaluamos era el valor por defecto de una función, no una propiedad de PASTIS.** Son 5 porque `build_spatial_kfold` trae 5, y el fold 5 tiene 176 celdas H3: admite muchos más. Al barrerlo pasa esto:
>
> | bloques | intervalo del contraste | ¿excluye el cero? | separación prueba–entrenamiento | clases estimables en el peor bloque |
> |---:|---|---|---:|---:|
> | **5** | (−0,041, **+0,008**) | **no** | **22,972 km** | **10 de 18** |
> | 15 | (−0,035, −0,015) | sí | 2,009 km | 2 de 18 |
>
> Con quince bloques **el resultado sale significativo**. Con cinco no.
>
> Mi propuesta es **quedarnos en cinco**, y quiero que sepas exactamente qué estamos comprando y qué estamos pagando.
>
> **Lo que pagamos**: el contraste principal deja de ser significativo. El artículo no puede decir «a igual cobertura gana la abstención». Pasa a decir «son equivalentes en calidad dentro de una banda declarada, y lo que de verdad cambia es a quién le retiran la promesa».
>
> **Por qué lo propongo igual**: con quince bloques los bloques son **vecinos** —2,0 km de separación, cuando el colchón que usamos para construirlos es de 1 km—, así que la independencia espacial que el artículo entero presume deja de estar demostrada. Y el peor bloque se queda con **dos** clases estimables de dieciocho, con lo que el F1-macro que promediamos ahí ya no es una media sobre el catálogo: es otra cosa con el mismo nombre.
>
> Y hay una razón que pesa más que las dos: **el quince es la celda más favorable de las siete que probé** —el mayor efecto y la menor dispersión a la vez—. Si lo elegimos, lo estamos eligiendo porque da significancia. Eso es exactamente lo que nos tumbó el artículo anterior, un piso más abajo.
>
> **La pregunta concreta, y quiero tu desacuerdo si lo tienes:**
>
> **¿Firmas que fijamos el número de bloques en cinco, por el criterio de separación espacial y clases estimables, sabiendo que ese es el valor donde nuestro contraste principal no alcanza significancia?**
>
> Si crees que hay un criterio mejor para elegirlo —uno que no mire el resultado—, dímelo ahora y lo aplicamos. Si crees que quince es defendible con esos 2,2 km, convénceme y lo discutimos. Lo que no podemos es decidirlo después de ver qué sale.
>
> De paso, tres cosas que van en el mismo ADR y son rápidas:
> 1. **Sede**: MICAI 2027 y solo MICAI, hasta veinte páginas. ¿De acuerdo, o prefieres reservar la mitad para una revista?
> 2. **El manuscrito de 24 páginas y el borrador retirado de 15**: ¿informe técnico interno o preprint? No pueden quedar en limbo.
> 3. **Isaac y Aaron**: hay que pedirles permiso por escrito para agradecerles, y eso solo lo puedes hacer tú. ¿Lo mandas esta semana?

---

## Lo que NO hay que preguntarle

Para no diluir la pregunta: el autor de correspondencia ya está decidido (él), los apellidos van con guion, y los cinco defectos de atribución bibliográfica ya están corregidos contra fuente. Nada de eso necesita su visto bueno.

## Lo que hay que decirle aunque no pregunte

- Su afirmación de que los checkpoints de segmentación se perdieron **era incorrecta**: están en el remoto DVC, a un `dvc pull` de 2,84 GB. Lo repetí sin comprobarlo y lo tengo corregido por escrito.
- De los tres checkpoints de FarSLIP que dio por perdidos, **uno es regenerable con receta verificada**, otro no se puede validar porque su configuración nunca entró al barrido, y **el tercero nunca existió**.
- El criterio de reentrenamiento que propuso, «~0,6452 macro-F1», **no existe en ningún artefacto**. El correcto es 0,7025.
