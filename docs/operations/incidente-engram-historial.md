# Incidente — memoria nativa de Engram en el historial público

**Estado**: contención y saneamiento **PENDIENTES DE AUTORIZACIÓN**. El diagnóstico y el ensayo
están hechos; los dos pasos irreversibles, no.
**Repositorio**: `jrebull/agrosat-copilotv2`, **público**, 0 forks.
**Origen**: `6247587` introduce `.engram/`; `f8984d8` sincroniza los chunks; `65e1f4d` los retira
del árbol. Retirarlos del árbol **no** los retira del repositorio.
**Referencia del procedimiento**: [GitHub — Removing sensitive data from a
repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

## Qué hay expuesto, medido y no estimado

Todo está en **un solo blob**, `.engram/chunks/7180e82c.jsonl.gz` (959 788 bytes descomprimido).
Los otros seis ficheros de `.engram/` no contienen ninguna de estas apariciones.

| Categoría | Apariciones | Detalle |
|---|---:|---|
| Direcciones de correo | 16 | 10 `gmail.com`, 3 `agrosat.dev`, 2 `tec.mx`, 1 cuenta de servicio `…iam.gserviceaccount.com` |
| Rutas `/Users/<usuario>` | 6 | un único nombre de usuario |
| Rutas `C:\Users\<usuario>` | 6 | |
| Rutas `/home/<usuario>` | 3 | |
| Identificadores tipo UUID | 6 | posibles identificadores de sesión |
| Direcciones IPv4 públicas | 8 | varias en rangos `34.x` / `35.x`, propios de Google Cloud |

**No hay credenciales.** Se buscaron, además de los patrones habituales de token, claves privadas
PEM, claves de AWS, JWT, cadenas de conexión y pares `campo = valor`. Los dos únicos hallazgos con
forma de credencial se clasificaron leyendo su contexto **sin volcar el valor**:

- Una URL `rediss://usuario:clave@<endpoint>:6379` cuyo **host es literalmente `<endpoint>`**: es
  la plantilla de documentación de `redis.asyncio.Redis.from_url(...)`, no una conexión real.
- Un `--secret=<nombre>` dentro de `gcloud secrets versions access latest --secret=… `: el valor
  capturado es el **nombre** del secreto, no el secreto.

Coincide con lo que encontró la revisión previa, con una red más ancha. Lo que sigue expuesto es
**información personal y de entorno**, no material criptográfico.

## Por qué los controles actuales no lo ven

`gitleaks` inspecciona los bytes del blob, y el blob está **comprimido con gzip**: los correos y
las rutas no aparecen en la forma que el escáner busca. Verde no significa limpio; significa que
no encontró sus patrones en lo que sabe leer.

## Alcance: quién sostiene los objetos

| Referencia | Estado | Quién puede retirarla |
|---|---|---|
| `refs/heads/main` | contiene los tres commits | la reescritura |
| `refs/heads/optimiza-harnees` | **rama fusionada que sigue viva**, con la punta exactamente en `f8984d8` | borrado normal |
| `refs/pull/3/head`, `refs/pull/4/head`, `refs/pull/5/head` | GitHub las conserva aunque se borre la rama | **solo GitHub Support** |

Esa última fila es la razón por la que la reescritura, por sí sola, no cierra el incidente.

## Ensayo hecho sobre un clon espejo, sin tocar nada

```
git clone --mirror <repo> ensayo.git
cd ensayo.git && git filter-repo --force --invert-paths --path .engram
```

Resultado medido:

- `.engram` desaparece de **toda** la historia; el blob de 959 KB deja de ser alcanzable.
- 539 → 538 commits: `f8984d8` solo tocaba `.engram`, así que queda vacío y se descarta.
- **El árbol de `main` es idéntico byte a byte** (`cff64062…` antes y después). La reescritura no
  cambia una sola línea del contenido de trabajo.
- `main` pasaría de `48259f1` a otro hash.

## Consecuencia que hay que planificar: la custodia

Reescribir cambia el hash de **48 commits**, y el ledger registra procedencia por SHA:

- **90 filas** de `paper/ARTIFACTS.md` llevan SHA de procedencia.
- **12** de esas quedarían apuntando a un commit inexistente.
- **78** sobreviven, por ser anteriores al primer commit reescrito.

Se arregla con `make paper-artifacts-seal`, que recalcula la columna desde git, **después** de la
reescritura y en un commit propio. Sin ese paso, `paper-artifacts-check` queda en rojo y parecería
que la custodia se rompió, cuando lo que cambió fue la numeración.

Igual de importante: los SHA citados en documentos y en el cuaderno público —mensajes de commit,
`docs/paper/respuesta-auditoria-externa.md`, la línea de revisión del sitio— dejarían de resolver.
No son custodia, pero son referencias que alguien seguirá.

## Procedimiento, en orden

1. **Contener**: repositorio a privado mientras dure la operación.
2. Avisar a Arthur **antes** de tocar nada: cualquier clon suyo queda divergente y no debe
   fusionarse después, hay que volver a clonar.
3. Borrar la rama `optimiza-harnees`, ya fusionada.
4. Reescribir sobre un espejo fresco del remoto y empujar todas las referencias con `--force`.
5. Pedir a GitHub Support la purga de `refs/pull/*` y de las vistas en caché.
6. Re-clonar en todas las máquinas. **Nadie** empuja desde un clon anterior.
7. `make paper-artifacts-seal`, y los nueve gates en verde.
8. Volver a público.

## Lo que no arregla ninguna reescritura

Los datos estuvieron publicados. Si alguien los copió, ya los tiene. La reescritura corta el
acceso futuro y reduce el daño; no lo revierte. Por eso el paso 2 es un aviso, no una consulta.
