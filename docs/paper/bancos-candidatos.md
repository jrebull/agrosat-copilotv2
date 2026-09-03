# Bancos candidatos para la validez externa (EPIC 22)

**Fecha**: 3 de septiembre de 2026. **Origen**: barrido bibliográfico verificado contra las páginas de cada depósito.
**Criterio de admisión**, en este orden: (1) partición espacial documentada, (2) licencia declarada y compatible, (3) descargable sin trámite, (4) fuera de Europa occidental.

## Recomendados

| Banco | Dónde | Licencia | Partición espacial | Por qué |
|---|---|---|---|---|
| **CropHarvest** | Zenodo `10.5281/zenodo.7257688`, 22,7 GB | CC BY-SA 4.0 | **Sí, explícita y documentada** | El único banco de tipo de cultivo fuera de Europa occidental descargable hoy con separación espacial declarada. Tres tareas: Kenia 1 345/45, Brasil 794/66, Togo 1 319/306 |
| **GEO-Bench `m-SA-crop-type`** | Zenodo `10.5281/zenodo.8008825`, 1,4 GB | CC BY 4.0 | Del propio GEO-Bench | Sudáfrica, nueve clases. Versión lista para ML de Spot the Crop, y conecta con GEO-Bench-2, que ya está en el stack del proyecto |

La frase de CropHarvest que responde por adelantado a la objeción de autocorrelación espacial: *«We removed the test polygons from the training sets to ensure there was no spatial overlap between the training and test sets, which could result in spatial autocorrelation.»*

## Descartados, con su motivo

| Banco | Motivo |
|---|---|
| CV4A Kenya | Entrenamiento y prueba **conviven en los mismos cuatro tiles**; sin partición espacial y sin licencia declarada |
| AgriFieldNet (India) | Licencia CC-BY-4.0 correcta, pero su documentación admite que *«some fields fall across multiple chips (in both train and test sets)»*: fuga espacial por diseño |
| Ghana y Sudán del Sur | Vía viva por SustainBench, pero su partición **solo preserva proporciones de clase, no es espacial** |
| SICKLE | No declara licencia en ninguna de sus páginas |
| Campo Verde y LEM (Brasil) | Acceso roto o con cuenta; conteos no verificables |
| Agriculture-Vision | Son nueve tipos de **anomalía de campo**, no tipo de cultivo |
| Hi-CNA, CropSAR, Sat2Farm | Sin evidencia de existir como bancos públicos de tipo de cultivo. **No citar** |
| México | Hueco confirmado por dos vías. No hay banco público por parcela |

## Notas operativas

- Radiant MLHub **migró**: `mlhub.earth/10.34911/rdnt.*` devuelve 301 a `source.coop/radiantearth/...`, y los DOI siguen resolviendo. El índice de Source Cooperative solo lista AgriFieldNet aunque los demás repositorios responden por URL directa, así que **no sirve para enumerar**.
- Los enlaces `registry.mlhub.earth` del repositorio `roserustowicz/crop-type-mapping` están rotos.

## Qué significa para el plan

**US-136** (tercer banco, fuera de Francia) → **CropHarvest**, tarea de Kenia o Togo.
**US-137** (cuarto banco, fuera de Europa occidental) → **GEO-Bench `m-SA-crop-type`** para Sudáfrica, o una segunda tarea de CropHarvest.

Los dos entran con la partición espacial que ya traen; no la construimos nosotros, que es justo la crítica que un revisor haría si la inventáramos.
