# ADR-015 — Engram como memoria compartida revisable

**Estado**: PROPUESTA; no autoriza sincronización ni versionado de memoria nativa.
**Fecha**: 5 de septiembre de 2026
**Proponente**: Arthur Jafed Zizumbo Velasco
**Decisor pendiente**: equipo mantenedor
**Incidente que la motiva**: PR integrado en `c214c28`

## Contexto

Engram puede conservar causas raíz, decisiones y aprendizajes entre sesiones y máquinas. Ese valor
es real: evita repetir investigaciones y hace visible el razonamiento que Git no contiene. También
tiene una frontera distinta a la documentación revisada. Un `engram sync` puede exportar prompts,
sesiones y observaciones completas; `scope: personal` no garantiza que una observación quede fuera
del conjunto del proyecto.

La primera integración convirtió esa posibilidad en obligación antes de acordar la política:
añadió chunks nativos a Git, importación automática y targets de sincronización. La revisión de los
payloads encontró datos personales y rutas locales. El gate buscaba patrones de tokens y proyectos
ajenos, pero no correos ni el principio más importante: solo debía entrar contenido permitido.

## Decisión provisional

Se conserva Engram como herramienta **local y opcional** de desarrollo. `.engram/` permanece
ignorado, no hay importación o exportación automática y ninguna memoria Engram es fuente normativa,
ledger del artículo ni memoria del producto. Las decisiones compartidas siguen viviendo en ADRs,
`docs/us-resolved/`, el plan y los registros ordinarios revisables.

La memoria compartida no se desecha: queda planificada detrás de los criterios siguientes. El
prototipo de unión de manifests puede mantenerse y probarse, pero no se conecta a `make check`, CI,
hooks ni al cierre de una US hasta que esta ADR sea ACEPTADA.

## Diseño que debe aprobarse

1. **Contenido por lista permitida.** Un esquema versionado admite solo decisión, causa raíz,
   consecuencia, puntero canónico y autor técnico. Todo campo libre adicional se rechaza.
2. **Redacción antes de exportar.** El exportador inspecciona el payload descomprimido, no el nombre
   ni el manifest. Rechaza correos, nombres propios no autorizados, tokens, UUID de sesión, rutas
   personales, geometrías de usuarios y cifras empíricas no publicadas.
3. **Revisión humana.** Cada lote produce una vista legible y un digest. Dos personas aprueban los
   mismos bytes antes de publicarlos; modificar el lote invalida las aprobaciones.
4. **Transporte y acceso.** El equipo elige explícitamente entre un repositorio privado dedicado o
   un artefacto cifrado con permisos nominales. La rama principal del código no es el valor por
   defecto.
5. **Retención y retirada.** Se fijan plazo, custodio, procedimiento de revocación y registro de qué
   clones recibieron cada lote. Borrar solo el archivo actual no se presenta como borrado del
   historial.
6. **Importación consciente.** La importación muestra proyecto, lote, autor, fecha y conteos antes
   de escribir la base local. No corre en `SessionStart` hasta que la política sea aceptada.
7. **Separación de autoridades.** Una memoria apunta a la fuente normativa; nunca reemplaza un ADR,
   el ledger, el preregistro, un artefacto o el plan.

## Pruebas de aceptación

- Un correo, token, UUID, ruta de usuario, resultado no publicado o proyecto ajeno hace fallar el
  gate aunque esté en `prompts`, `sessions`, `mutations` u otro camino lateral.
- Un campo desconocido hace fallar el gate: se enumera lo permitido, no solo lo prohibido.
- Un lote alterado después de la revisión rompe el digest y exige nuevas aprobaciones.
- Un archivo nuevo fuera del manifest, una entrada sin archivo y dos rutas para el mismo lote
  fallan de forma distinguible.
- La ausencia de Engram, del plugin o de la red nunca impide ejecutar, probar o cerrar una US.
- Un checkout nuevo no importa memoria sin una acción explícita del desarrollador.
- La prueba usa fixtures sintéticos; no depende de los chunks reales ni de una base personal.

## Condición para aceptar esta ADR

El equipo debe elegir transporte, custodio, personas autorizadas y retención; implementar el
exportador por lista permitida y sus pruebas negativas; y revisar un lote piloto sintético. Solo
entonces se podrán añadir targets con nombres explícitos de experimento. `make memory-sync`, los
hooks automáticos y el versionado de `.engram/` continúan prohibidos mientras el estado sea
PROPUESTA.

## Consecuencias

- Se conserva la idea de Arthur y se convierte en trabajo verificable, no en una capacidad fingida.
- Se retiran del árbol actual los chunks nativos ya publicados. Permanecen en el historial Git del
  commit original; purgar ese historial sería una operación separada y coordinada.
- Engram local sigue disponible para quien lo instale, dentro de la frontera descrita por
  `.claude/skills/agrosat-engram-memory/SKILL.md`.
- El artículo MICAI y US-172 no dependen de esta ADR.
