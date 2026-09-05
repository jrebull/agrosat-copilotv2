# ARTIFACTS — ledger de custodia del manuscrito MICAI 2027

Cada cifra impresa en el artículo debe re-derivarse desde un archivo de esta tabla.
Lo que no aparece aquí no se imprime. El gate `make paper-artifacts-check` recalcula
el MD5 de cada fila sellada y falla si algo cambió sin registrarse.

**Sellado el**: 2026-09-04 · **Commit de sellado**: `8ab77bd`
> El gate comprueba que ese commit exista y este en la historia de HEAD, e imprime cuantos
> commits ha quedado atras. No se exige que sea HEAD: el commit que actualiza el ledger no
> puede conocer su propio sha.
**Fase**: 6 de [`docs/plan-micai-2027.md`](../docs/plan-micai-2027.md) ·
**Ángulo vigente**: [`ADR-013`](../docs/decisions/ADR-013-angulo-micai.md) (reencuadre).

## Aviso de vigencia: lo que produjo el módulo con los tres defectos

**Todo artefacto derivado de `ml/eval/paper_micai_coverage.py` sellado antes del 3 de septiembre
de 2026 es EXPLORATORIO y está pendiente de regenerar.** El sello sigue siendo válido —los bytes
son los que dice— pero las cifras salieron de una versión con tres defectos que una auditoría
externa encontró leyendo el código:

1. el universo de clases del macro salía de las verdades de las parcelas **entregadas**, que
   dependen del mecanismo, así que el denominador se movía entre brazos;
2. el umbral del rechazo por confianza se elegía **dentro del bloque que lo mide**;
3. el intervalo remuestreaba **parcelas** dentro de cada bloque, convirtiendo cinco bloques
   espaciales en dieciséis mil réplicas que no existen.

Los tres están reparados en el código, con tests que fallan sobre la versión anterior
(`tests/ml/eval/test_paper_micai_coverage.py`). Regenerar los artefactos es el paso de artefacto
de US-124 y US-125, y hasta entonces **ninguna de estas cifras entra en el artículo**:

| Prefijo | Qué contiene |
|---|---|
| `reports/paper_micai/fase3/*` | La frontera del banco primario |
| `reports/paper_micai/fase4/replica_*` | La réplica en BreizhCrops |
| `reports/paper_micai/diagnostico/*` | El coste medido de los tres defectos (este mide el defecto a propósito y se conserva tal cual) |
| `reports/paper_micai/potencia/*` | La potencia de las cuatro medidas de disparidad |
| `reports/paper_micai/bloques/*` | El barrido de k |
| `reports/paper_micai/equidad/*` | La cobertura por clase |

Los artefactos de `reports/paper_micai/prereg/*` **no** dependen de ese módulo: miden geometría
del diseño y siguen vigentes.

## Versiones de cómputo del sellado

| Paquete | Versión |
|---|---|
| `xgboost` | 3.4.1 |
| `scikit-learn` | 1.9.0 |
| `polars` | 1.44.1 |
| `matplotlib` | 3.11.1 |
| `numpy` | 2.3.5 |
| `pandas` | 2.3.3 |

## Artefactos sellados

| Elemento del paper | Artefacto | MD5 | Bytes | Commit | Estado | Nota |
|---|---|---|---|---|---|---|
| Miembros y protocolo: manifiesto histórico del arnés OOF (fold 5, 18 clases, una corrida) | `ml/eval/oof/manifest.json` | `e0a2e793f12dd1d4879e7fad4ce1597c` | 3214 | `52f66b8` | SELLADO | Manifiesto v1 de una sola corrida de seis modelos, con `code_version=086c4b...`. Se retiró la fila de Full-M que se había pegado desde otra corrida con otro código: su procedencia vive en el inventario global y en su informe sellado. Los manifiestos v2 nuevos llevan `code_version`/`data_version` por entrada y, para temporales, `n_timesteps_dataset` y `n_timesteps_model_spec`; el gate exige el acoplamiento. |
| Universo del ensamble: posteriores por parcela, U-Net | `ml/eval/oof/oof_parcel_unet_fold5.parquet` | `ff36268a0d9bdbaa732374fa031fcf3f` | 2015608 | `23cb470` (.dvc) | SELLADO | Miembro denso espacial. |
| Universo del ensamble: posteriores por parcela, DeepLabv3+ | `ml/eval/oof/oof_parcel_deeplabv3plus_fold5.parquet` | `6a2b13aa4213b73349310f767934635c` | 2075019 | `23cb470` (.dvc) | SELLADO | Miembro denso espacial. |
| Universo del ensamble: posteriores por parcela, SegFormer | `ml/eval/oof/oof_parcel_segformer_fold5.parquet` | `99d8b9c0e766d1a42f1cb6543bb7fba6` | 2062477 | `23cb470` (.dvc) | SELLADO | Miembro denso espacial. |
| Universo del ensamble: posteriores por parcela, U-TAE | `ml/eval/oof/oof_parcel_utae_fold5.parquet` | `597d867f32f9f7c52ff75702e2cf1c14` | 2058667 | `23cb470` (.dvc) | SELLADO | Bytes canónicos; US-119 lo excluye del panel inferencial mientras no se identifique la causa de su caída. Se conserva como descriptivo. |
| Universo del ensamble: posteriores por parcela, TSViT-pheno | `ml/eval/oof/oof_parcel_tsvit-pheno_fold5.parquet` | `b87f9b63ccecf4e5395bb22a307d67e8` | 2071454 | `23cb470` (.dvc) | SELLADO | Miembro del universo campeón. |
| Universo del ensamble: posteriores por parcela, TSViT-pheno-fullm | `ml/eval/oof/oof_parcel_tsvit-pheno-fullm_fold5.parquet` | `491df530cc03f34cd68268987758f76a` | 2072878 | `373acb7` (.dvc) | SELLADO | **Regenerado por el PR #1 (`373acb7`) con el fix de `n_timesteps`.** El fichero anterior salia de un volcado que alimentaba al dataset con T=10 mientras el modelo se reconstruia con T=37: F1-macro 0,2552 en vez de 0,7883. Reproducido de forma exacta desde `CHECKPOINT_REGISTRY` tras el fix, argmax 1,0000 en los cuatro estratos de confianza. |
| Universo del ensamble: posteriores por parcela, AnySat | `ml/eval/oof/oof_parcel_anysat_fold5.parquet` | `3366dc2469ab82575e82f2d8bfe0ffcc` | 2036192 | `23cb470` (.dvc) | SELLADO | Bytes canónicos; US-119 lo excluye del panel inferencial mientras no se identifique la causa de su caída. Se conserva como descriptivo. |
| Universo del ensamble: posteriores por parcela, XGBoost sobre AlphaEarth | `ml/eval/oof/oof_parcel_xgb-alphaearth_fold5.parquet` | `97a083d90b1635dbfc216d837325877f` | 1192508 | `10084d2` (.dvc) | SELLADO | Artefacto histórico `legacy_unverified`: no define la población ni puede entrar en inferencia. La reconstrucción con identidad propia es `xgb-alphaearth-remat-v1`. |
| Universo del ensamble: posteriores por parcela, FarSLIP fine-tuned 18 | `ml/eval/oof/oof_parcel_farslip-ft18_fold5.parquet` | `433325d8ad2cb7c65a26c19065f25295` | 1150480 | `1890bef` (.dvc) | SELLADO | Artefacto histórico conservado; el inventario lo marca `excluded` para MICAI 2027. |
| Universo del ensamble: posteriores por parcela, FarSLIP zero-shot | `ml/eval/oof/oof_parcel_farslip-zeroshot_fold5.parquet` | `28c2b44de3fd13e4cadb3937c5bcbea0` | 1075770 | `1890bef` (.dvc) | SELLADO | Artefacto histórico conservado; el inventario lo marca `excluded` para MICAI 2027. |
| Rejilla de ensambles: Stacking-3, Stacking-5, Blending-3 y Blending-5 en dos universos | `reports/ensemble/metrics/us043_farslip_grid.csv` | `b0bddf7a3515717d84dd9107592975bb` | 844 | `1890bef` | SELLADO | Fuente única de 0,7486 / 0,8495 (Stacking-5, universo tsvit-pheno) y de 0,7470 (Stacking-3). |
| Aporte de FarSLIP y deltas 5 frente a 3 miembros | `reports/ensemble/us043_farslip_summary.json` | `ff55c73cdba9048181eb8e3ef93326d9` | 870 | `1890bef` | SELLADO | Cuidado: las claves dicen 'oof_cv' pero el régimen es fold-5 held-out sobre el universo tsvit-pheno-fullm. |
| Rejilla de stacking y blending por universo | `reports/ensemble/metrics/us043_farslip_stacking_blending.csv` | `4cd6ef6e8778e092ba842105c6789891` | 581 | `1890bef` | SELLADO | Detalle de la rejilla anterior. |
| Curva de retirada honesta de clases: F1-macro y cobertura de 18 a 8 clases | `reports/ensemble/metrics/us043_honest_dropout_curve.csv` | `c08df81d0eccc4835d684536a2761407` | 1038 | `9ee18b7` | SELLADO | Núcleo de la contribución central: 18 clases 0,7486 con 16 640 parcelas; 9 clases 0,9121 con 13 624. |
| Curva de cardinalidad por soporte acumulado | `reports/ensemble/metrics/us043_winner_cardinality_curve.csv` | `f46bfcc6c98b105130331f1f563b59f0` | 745 | `1890bef` | SELLADO | Eje distinto al anterior: top-k por soporte, no retirada de clases. No confundir. |
| Desempeño por clase del campeón | `reports/ensemble/metrics/us043_winner_per_class.csv` | `72fdb7179a5b840a4916ab78f5b09279` | 2095 | `1890bef` | SELLADO | Base del delta por clase de la fase 2. |
| Nulo de vecindad: barrido k x alfa con veredicto | `reports/ensemble/metrics/ec_neighborhood_result.json` | `1824155553a74b10ce387c6e04891002` | 5501 | `2c8dc2b` | SELLADO | Da +0,0027 en el punto que mejora ambos ejes y +0,0068 en el óptimo de 18 clases; umbral 0,01; sin intervalo. |
| Los cuatro combinadores con SUS DOS regímenes en columnas contiguas | `reports/ensemble/metrics/weighted_voting_pastis.csv` | `e35eb0fc7ef375906d42dc4813f4091c` | 355 | `03b41f4` | SELLADO | Prueba de que la cifra libre de fuga existía: f1_macro junto a f1_macro_spatialcv. La fila del voto simple 1/N, que no ajusta nada, separa el coste de la fuga del coste de agregar por bloque. |
| Comparativa de los cuatro ensambles de rúbrica | `reports/ensemble/metrics/comparison_us040.csv` | `76817f5ee3ca330418e15c03540aa756` | 429 | `10084d2` | SELLADO | Origen de la cifra 0,7470 del Stacking-3. |
| FarSLIP fiel frente a AlphaEarth en sonda de parcelas | `reports/farslip/metrics/us037_farslip_fiel_vs_alphaearth.csv` | `6b99e9ccfaf28625a6c21fdc040c3005` | 243 | `2625188` | SELLADO | Contiene tanto el eje de F1 como el de silhouette; no mezclarlos. |
| Recableo del perceptor del copiloto al campeón | `reports/agent_bench/perceiver_champion_eval_v2.json` | `04ae7461ee3df7a2d155006ec09c2747` | 1755 | `48dd6cf` | SELLADO | Solo si la capa conversacional entra al texto. |
| Transferencia densa Francia a Cataluña | `reports/segmentation/sen4agrinet_transfer_result.json` | `06125c50fd921a936aeb4069daaf2ee8` | 736 | `47b4000` | SELLADO | Fuera del cuerpo bajo el ángulo reencuadrado. |
| Ablación de arranque en caliente de DE4 | `reports/us079_figs/ablation_compare.json` | `3e10a62e00f407c16edfcdb3673de1f8` | 158701 | `b4c82db` | SELLADO | Fuera del cuerpo; el resto de DE4 no tiene artefacto. |
| Curva few-shot de EuroCropsML LV a EE (3 semillas, k de 1 a 500) | `data/transfer/eurocropsml_fewshot_results.parquet` | `4eb4dfa0ffc530f8501c1815e6959db2` | 3394 | `bc019e5` | SELLADO | 63 filas reales. `pendientes-arthur.md` lo daba por perdido: existe y no estaba en DVC. |
| Embeddings WorldCereal de Brasil (Cerrado) | `data/transfer/worldcereal_brazil_cerrado.parquet` | `aca0ab07c4f5ed20fecb6fe2ea81c44a` | 215036 | `a50cb94` (.dvc) | SELLADO | Insumo crudo del barrido tropical; el resultado del barrido no existe. |
| Embeddings WorldCereal de India (Karnataka) | `data/transfer/worldcereal_india_karnataka.parquet` | `f6fb984f62f939e50b6048e9a74aa0cb` | 201319 | `a50cb94` (.dvc) | SELLADO | Insumo crudo del barrido tropical; el resultado del barrido no existe. |
| Deltas pareados multirregión | `data/transfer/multiregion_paired_delta.parquet` | `608a994e120e74ace9d5eaf4923d46d4` | 3657 | `a50cb94` (.dvc) | SELLADO | Fuera del cuerpo bajo el ángulo reencuadrado. |
| Macro por clase multirregión | `data/transfer/multiregion_per_class_macro.parquet` | `51496ae0321af2b29ee8cc6edb3642a2` | 2549 | `a50cb94` (.dvc) | SELLADO | Fuera del cuerpo bajo el ángulo reencuadrado. |
| Etiquetas por parcela del fold 5, población elegible independiente de los predictores | `reports/paper_micai/fase1/parcel_gt_fold5.parquet` | `dddb1ea9c1b0b15c0f24142e8dd0ec4d` | 68147 | `9eaeb2f` | SELLADO | 16 640 filas derivadas de `metadata.geojson`: `Fold=5` y etiqueta `semantic18` válida. Sus bytes no cambiaron al corregir la derivación. Permite reproducir toda la evaluación sin los 68 GB del dataset. |
| Centroides por parcela del fold 5 | `reports/paper_micai/fase1/parcel_centroids_fold5.parquet` | `5b1ce7b92059e62f640869a51bc8f1e6` | 358870 | `9eaeb2f` | SELLADO | Geometrías necesarias para los sub-folds espaciales del meta-modelo y para el nulo de vecindad. |
| Soporte por clase del fold 5 | `reports/paper_micai/fase1/parcel_gt_fold5_support.csv` | `e9c3b084ad2f8dba43e241f5ff607e46` | 136 | `262a519` | SELLADO | El desbalance del que trata el artículo: de 6 128 parcelas en la clase mayoritaria a 103 en la menor. |
| Procedencia del ground truth del fold 5 | `reports/paper_micai/fase1/parcel_gt_fold5_provenance.json` | `8b5bedf907c603affd0f5840f405434f` | 1104 | `57039f6` | SELLADO | La poblacion es la ELEGIBILIDAD DEL BANCO —parches con Fold == 5 en metadata.geojson y etiqueta semantic18 valida—, no la interseccion de los miembros OOF. **Al regenerarla, las etiquetas, los centroides y el soporte salieron BYTE A BYTE IDENTICOS**: la interseccion coincidia con la elegibilidad, asi que el cambio es de dependencia y no de poblacion. Lo que cambia es este fichero, que ahora lo dice.|
| Fase 1: cobertura de cada miembro sobre la poblacion elegible | `reports/paper_micai/fase1/cobertura_por_miembro.csv` | `7b236c01ab5cfb439ec9a5ecd230f42b` | 315 | `57039f6` | SELLADO | Ocho miembros canonicos, todos con cobertura 1,0 sobre las 16 640 parcelas elegibles. Los 11 892 'fuera de la poblacion' de los miembros densos son parcelas sin etiqueta semantic18 valida. La cobertura pasa a ser un DATO que se reporta y no un filtro que define la poblacion. |
| Fase 2: los diez miembros bajo un protocolo único | `reports/paper_micai/fase2/individuales_protocolo_unico.csv` | `59d4c8bfffb0505da27caaf644389e6c` | 626 | `87cccdd` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. El mejor individual es tsvit-pheno con F1-macro 0,7367; ninguna combinación libre de fuga lo supera.|
| Fase 2: las cuatro reglas de combinación con su régimen | `reports/paper_micai/fase2/combinaciones.csv` | `036178fa87848b74bc3e5a190b5f96d1` | 483 | `87cccdd` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. Separa el refit sobre todo (in-sample) del agrupado espacial (libre de fuga); ahí se ve el origen del 0,7486.|
| Fase 2: bootstrap pareado y McNemar del arbitraje | `reports/paper_micai/fase2/arbitraje_pruebas.json` | `9c2191f7ed5537857b7bdb5b22ae6a11` | 2223 | `87cccdd` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. El árbitro no gana al promedio y pierde contra el mejor individual, con intervalos que excluyen el cero.|
| Fase 2: F1 por clase de las cuatro reglas | `reports/paper_micai/fase2/arbitraje_por_clase.csv` | `18493e9b47b5b00d60fe5cb664205d51` | 1556 | `87cccdd` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. Muestra que el árbitro retira la clase 10 en silencio: F1 exactamente 0 sobre 355 parcelas.|
| Fase 2: posteriores del árbitro agrupado libre de fuga | `reports/paper_micai/fase2/arbitro_agrupado_posteriores.parquet` | `f0747c3a71108d4fb1c183b40bb2f5a4` | 2325843 | `87cccdd` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. Cada parcela predicha por el bloque que no la vio; insumo del nulo de vecindad.|
| Fase 2: cobertura y calidad por bloque y mecanismo | `reports/paper_micai/fase2/cobertura_por_bloque.csv` | `c5e9d5c7f202c764958e243b591ae63b` | 5757 | `8be4f90` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. Cada bloque contra UNA leyenda elegida fuera de él, con la leyenda registrada.|
| Fase 2: resumen de la frontera calidad-cobertura | `reports/paper_micai/fase2/cobertura_resumen.csv` | `f2c73a3c79519829f8b4c718bbed2bc1` | 1327 | `8be4f90` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. Media, mínimo y máximo entre bloques de los dos mecanismos a igual cobertura.|
| Fase 2: comparativa densa por fold de las variantes de TSViT | `reports/segmentation/metrics/tsvit_pheno_vs_base_fold5.csv` | `44b73e188a1a510e805b062d8d62facc` | 588 | `71ca906` | SELLADO | Da 0,7918 de F1-macro pixel a tsvit-pheno-fullm, frente a 0,2552 de su volcado por parcela: la inconsistencia B2 de la auditoria. |
| Fase 2: contraste de los dos mecanismos con intervalo | `reports/paper_micai/fase2/cobertura_comparacion.json` | `53412f216cfc831e1631f36543733efa` | 4188 | `8be4f90` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. Contribución central: retirar clases domina al rechazo por confianza y el IC excluye el cero por debajo de doce clases.|
| Fase 2: barrido completo del refinamiento por vecindad | `reports/paper_micai/fase2/vecindad_barrido.csv` | `4b6399f6c8a9da817e71cee396e6666e` | 1484 | `18e699e` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. Ningun punto con alfa mayor que cero mejora al alfa cero sobre el mejor predictor libre de fuga.|
| Fase 2: veredicto del nulo de vecindad con intervalo | `reports/paper_micai/fase2/vecindad_veredicto.json` | `568ffd6739409dda52274b6d0d7c3f14` | 1710 | `18e699e` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. Regla R1 de ADR-013: el intervalo incluye el cero, asi que es un nulo acotado.|
| Fase 2: FarSLIP cinco frente a tres miembros | `reports/paper_micai/fase2/farslip_cinco_vs_tres.csv` | `0a83f40f2279cf4c6a19bccaacfad024` | 568 | `18e699e` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. Las filas in-sample reproducen 0,7470 y 0,7486; las libres de fuga dan 0,6789 y 0,6794.|
| Fase 2: delta de FarSLIP con intervalo | `reports/paper_micai/fase2/farslip_delta.json` | `b6fe5ac82c3da95cfe339afaa70cdfbc` | 1231 | `18e699e` | OBSOLETO | **OBSOLETO desde el 4 de septiembre de 2026**: se calculo con los diez miembros de ALL_MEMBERS, y tres de ellos —farslip-ft18, farslip-zeroshot y el historico xgb-alphaearth— han pasado a `excluded` o `legacy_unverified` en `ml/eval/oof/inventario.json`. El sello acredita sus bytes, no que sus miembros sigan admitidos. Se regenera cuando el panel este fijado. +0,0006 con IC de [-0,0024, +0,0034]: el aporte no se distingue de cero fuera del regimen in-sample.|
| Fase 3: la frontera rehecha, por bloque y mecanismo | `reports/paper_micai/fase3/frontera_por_bloque.csv` | `8a10691f779c155067b2e06d60bd0874` | 30960 | `686621d` | OBSOLETO | Cuatro series por predictor con estimando alineado y entrega sin oráculo; incluye la leyenda de cada bloque. |
| Fase 3: resumen de la frontera | `reports/paper_micai/fase3/frontera_resumen.csv` | `cf1dce117ca1d6eeacb595596d46b469` | 5808 | `686621d` | OBSOLETO | Media, mínimo y máximo entre bloques del F1 alineado, y el F1 nativo al lado, que NO es comparable entre mecanismos. |
| Fase 3: contrastes con intervalo pareado y corrección de Holm | `reports/paper_micai/fase3/frontera_contrastes.json` | `bc282a67b8b69b71e6037c1e9748e0e4` | 15532 | `686621d` | OBSOLETO | H1 no se sostiene: a K=9 el IC incluye el cero en los dos predictores y ningún contraste sobrevive a Holm. |
| Fase 3: figura de la frontera | `reports/paper_micai/fase3/frontera.svg` | `d103c5ad7c6a1209f7dab329915e5a97` | 57355 | `a35e96f` | OBSOLETO | Tres mecanismos más el control sin mecanismo, con el criterio principal marcado. |
| Fase 4: caracteristicas por parcela de BreizhCrops (185 columnas, 60 000 parcelas) | `reports/paper_micai/fase4/breizhcrops_features.parquet` | `274a4cc93c79fce8b797e6671cdf1e3d` | 55923894 | `2d96ca7` (.dvc) | SELLADO | Submuestreo proporcional de 30 000 por region, semilla 42, mismas 185 caracteristicas que el conjunto primario. |
| Fase 4: soporte por clase y region | `reports/paper_micai/fase4/breizhcrops_soporte.csv` | `026f4d8698cb48862f0108e0a3f9a98b` | 430 | `2d96ca7` | SELLADO | Siete clases por encima de cien parcelas; girasol con 2 y nueces con 7 en total. |
| Fase 4: procedencia del sellado de caracteristicas | `reports/paper_micai/fase4/breizhcrops_procedencia.json` | `517df9be473300707126e8ebbdc7662d` | 510 | `2d96ca7` | SELLADO | Semilla, submuestreo, version de codigo y el conteo real de 185 caracteristicas. |
| Fase 4: posteriores dejando una region fuera | `reports/paper_micai/fase4/breizhcrops_posteriores.parquet` | `bcc55122c8047c75f644d852c5880b11` | 2593890 | `2d96ca7` (.dvc) | SELLADO | Cada posterior viene de un modelo que no vio la region de esa parcela. |
| Fase 4: replica del protocolo, por bloque y mecanismo | `reports/paper_micai/fase4/replica_por_bloque.csv` | `d85baf968e38869e3437b94283a8c536` | 9955 | `2d96ca7` | OBSOLETO | Cuatro series en los dos universos declarados, con la leyenda y la cobertura de cada bloque. |
| Fase 4: contrastes de la replica con Holm | `reports/paper_micai/fase4/replica_contrastes.json` | `d023732d961bfe2d9d04ba10849259fd` | 11802 | `2d96ca7` | OBSOLETO | La descomposicion se transporta: 95,1 % denominador en el criterio principal y 100 % donde la cobertura sigue completa. |
| Fase 4: figura de la replica | `reports/paper_micai/fase4/replica.svg` | `3f54d20d9c92505866cef87a2448219d` | 55182 | `a35e96f` | OBSOLETO | Los dos universos lado a lado; las cuatro series coinciden hasta K = 6. |
| Diagnostico: coste de los tres defectos de protocolo | `reports/paper_micai/diagnostico/diagnostico_protocolo.json` | `cf997ef2e32bf23477cf6c0f7c461d18` | 2522 | `7c69551` | SELLADO | El 42 por ciento, el intervalo por bloques y la ventaja de la asimetria, que se habian publicado en prosa sin artefacto. |
| Diagnostico: el denominador, bloque a bloque | `reports/paper_micai/diagnostico/denominador_por_bloque.csv` | `9a643b3c6be5e1523b35cd6ac539c82a` | 238 | `7c69551` | SELLADO | Cinco filas; el bloque 2 cambia de signo al igualar el denominador. |
| Potencia del criterio principal nuevo, con cuatro medidas declaradas | `reports/paper_micai/potencia/potencia_disparidad.json` | `028461a1bbd4dc5c87969597233e7b24` | 1764 | `50498ea` | OBSOLETO | Ninguna de las cuatro medidas de disparidad tiene potencia con cinco bloques: hacen falta entre doce y diecisiete. |
| Potencia: la disparidad bloque a bloque | `reports/paper_micai/potencia/disparidad_por_bloque.csv` | `28fba97ed0deff7b9dba33127cff25cf` | 856 | `50498ea` | OBSOLETO | Cuatro medidas por bloque y mecanismo, con soporte minimo de cincuenta parcelas por clase. |
| Barrido del numero de bloques espaciales | `reports/paper_micai/bloques/bloques.json` | `f5e2fd620edcb05a417a0e3fd42671b3` | 2965 | `50498ea` | OBSOLETO | Cinco bloques era el valor por defecto, no una restriccion del dato. **Su columna `tiene_potencia` NO se cita**: compara el efecto con un MDE calculado tratando las subdivisiones como n replicas independientes, y no lo son — el Jaccard entre entrenamientos sube de 0,43 con k=5 a 0,62 con k=25. Partir mas fino baja el MDE por aritmetica, no por informacion. Se rehace con la unidad del estimando de US-173. Y elegir k por su resultado es p-hacking. |
| Barrido de bloques, tabla | `reports/paper_micai/bloques/barrido_bloques.csv` | `bef2b85ceb9f3ed5507bd547180ec7fb` | 598 | `50498ea` | OBSOLETO | Siete valores de k con su delta, su intervalo y su MDE **aproximado** (t central, y con la unidad equivocada: ver la fila de arriba). Lo que se cita de esta tabla es la SENSIBILIDAD del estimador a k, que es el resultado; no la columna de potencia. |
| Preregistro: parametros del diseno, medidos antes de fijarlos | `reports/paper_micai/prereg/parametros_prereg.json` | `1ca981af6b864530add66e36ce812ec6` | 13642 | `e2a61aa` | SELLADO | Separacion ENTRE CENTROIDES exacta (KD-tree) —cota superior de la separacion entre parcelas, no demuestra independencia— y clases estimables por k, barrido de colchon, solapamiento entre entrenamientos y universo de clases por bloque. **Ya NO produce banda de equivalencia**: la anterior estaba anclada en el MDE y la retiro la auditoria externa. NO calcula ningun contraste entre mecanismos. |
| Preregistro: tabla del diseno por k | `reports/paper_micai/prereg/parametros_diseno.csv` | `0c473f38a61eeea8e49ea72d55306011` | 803 | `e2a61aa` | SELLADO | El criterio elegido selecciona k=5: separacion minima **exacta entre centroides** de 22,972 km frente a 1,975 desde k=8. Entre centroides, no entre limites de parcela: es una cota superior y por si sola no demuestra independencia, y diez clases estimables en el peor bloque frente a dos desde k=12. Las cifras anteriores (23,505 y 2,877) venian de submuestrear ~300 puntos por lado antes de tomar el minimo, lo que sesga hacia arriba por construccion. |
| Preregistro: barrido de colchon | `reports/paper_micai/prereg/barrido_colchon.csv` | `912ee4e7db91d060605674f1616dc1a2` | 572 | `e2a61aa` | SELLADO | Veinte combinaciones de k y colchon con su separacion exacta entre centroides, sus clases estimables y las parcelas de prueba que sobreviven. Estos numeros estaban SOLO en la prosa del preregistro; ahora tienen productor. Con k=5 el colchon no excluye nada porque los bloques ya estan a 23 km. |
| Consulta etica US-172: protocolo | `reports/us172/protocolo-US172-v0.2-7f384df.pdf` | `05c0310e48118bc31b6e8fbde483276a` | 62827 | sin seguimiento en git (`*.pdf` global) | SELLADO | Protocolo completo v0.2. Lleva en pie de pagina el commit del que sale, asi que la fuente se recupera byte a byte. `make us172-adjuntos` lo reconstruye byte a byte con SOURCE_DATE_EPOCH fijo. |
| Consulta etica US-172: consentimiento | `reports/us172/consentimiento-US172-v0.2.pdf` | `30b5d7545726bb81ea843cf63a0089f8` | 22213 | sin seguimiento en git (`*.pdf` global) | SELLADO | Anexo A, para el comite y para los informantes. `make us172-adjuntos` lo reconstruye byte a byte con SOURCE_DATE_EPOCH fijo. |
| Consulta etica US-172: filtro-elegibilidad | `reports/us172/filtro-elegibilidad-US172-v0.2.pdf` | `f382262eed063d875db4c1c4d8b0c8fd` | 21785 | sin seguimiento en git (`*.pdf` global) | SELLADO | Anexo D, con registro de todos los contactados incluidos excluidos y declinantes. `make us172-adjuntos` lo reconstruye byte a byte con SOURCE_DATE_EPOCH fijo. |
| Consulta etica US-172: plan-custodia | `reports/us172/plan-custodia-US172-v0.2.pdf` | `de291133edba7fb202ece7a8071a8d1b` | 32719 | sin seguimiento en git (`*.pdf` global) | SELLADO | Apartados 6, 9 y 10 bis: custodio, cifrado, permisos, retencion, destruccion, vinculo reversible y campos operativos pendientes. `make us172-adjuntos` lo reconstruye byte a byte con SOURCE_DATE_EPOCH fijo. |
| Verificacion OOF: tsvit-pheno-fullm | `reports/paper_micai/oof/verificacion-tsvit-pheno-fullm.json` | `d9107245151fe1f9a741007a56db0a9d` | 1507 | `1dd6a41` | SELLADO | Verificacion contra el artefacto CORREGIDO: n_timesteps 37 en la especificación del modelo y en el dataset, 496 parches, 28 532 posteriores, argmax 1,0000 en los cuatro estratos y diferencia media 0,000000. El campo histórico `n_timesteps_checkpoint` del JSON toma el valor del registro, no de metadatos internos del checkpoint; el productor futuro lo llama `n_timesteps_model_spec`. |
| Verificacion OOF: xgb-alphaearth-remat-v1 | `reports/paper_micai/oof/verificacion-xgb-alphaearth-remat-v1.json` | `930db7a73ebffa1439179aae0723d91c` | 1091 | `574ae74` | SELLADO | Comparacion de la re-materializacion con el historico: argmax 0,9454, y por estratos 0,5474 por debajo de 0,5 de confianza frente a 0,9999 por encima de 0,9. El desacuerdo esta donde el modelo duda, que es la franja sobre la que opera el articulo. |
| Sanidad de miembros: sanidad_miembros.csv | `reports/paper_micai/us119/sanidad_miembros.csv` | `03b46366c18881cdbd9ee312231e03c3` | 2930 | `15eb91c` | SELLADO | US-119: metrica declarada por cada checkpoint frente a la de su volcado sobre fold 5, las dos POR PIXEL, con el F1 por parcela en columna aparte. Cinco miembros dentro del umbral de 0,15; anysat (0,3597) y utae (0,4010) fuera. Cada fila lleva la decisión y la identidad de los pesos. |
| Sanidad de miembros: sanidad_miembros.json | `reports/paper_micai/us119/sanidad_miembros.json` | `b4d3aee978d35c7ecdb4a8cddba56d30` | 8215 | `15eb91c` | SELLADO | Procedencia ejecutable: rutas y SHA-256 de los checkpoints, conteos tensor a tensor —380 U-Net, 501 AnySat, 208 SegFormer—, búsqueda recursiva de folds, decisión por miembro y resumen de cinco incluidos, dos excluidos y la premisa desmentida por SegFormer. |
| Equidad: cobertura por clase bajo cada mecanismo | `reports/paper_micai/equidad/cobertura_por_clase.csv` | `a254affe1425a0e5a55622e9c09bf601` | 1115 | `9e6fc19` | OBSOLETO | Quien paga la abstencion, medido. Responde a la objecion de que citabamos a Jones et al. al reves. |
| Equidad: correlaciones y clases sin cobertura | `reports/paper_micai/equidad/equidad.json` | `1ddb39e59fd5990ab3ffd2e342620df1` | 1303 | `9e6fc19` | OBSOLETO | Spearman entre soporte y cobertura por mecanismo y banco, con su p. |
| Fase 6: manuscrito MICAI, fuente principal | `paper/micai/main.tex` | `b603d1ba9600aa537cfc8710a2c236ce` | 4388 | `6fa433d` | SELLADO | Clase llncs con `\newif\ifanon`; anonimo por defecto, A4 real, `hidelinks`. |
| Trabajo relacionado verificado en fuente primaria | `docs/paper/trabajo-relacionado-verificado.md` | `51a8015d81154d4cbae6ab02c053b378` | 12964 | `2371eff` | SELLADO | NORMATIVO: fija, referencia por referencia, que afirmacion sostiene, cual es su limite metodologico, y la redaccion PERMITIDA y PROHIBIDA. Dieciseis claves, todas en refs-candidates.bib. Incluye el alcance de la busqueda con sus sesgos —entre ellos el que hizo que el precedente mas cercano no apareciera por vocabulario—, la formula de novedad limitada sin declarar exhaustividad, y el parrafo de primera pagina rotulado como borrador no integrado. |
| Catalogo verificado de referencias del manuscrito NUEVO | `paper/micai2027/refs-candidates.bib` | `080602404bac5be8476fd777e863584d` | 23868 | `cfc07f2` | SELLADO | 57 entradas: las 44 de la matriz verificada mas Ha 1997, Mortier 2021 y Chzhen 2021, que US-141 exige y no estaban. **Fila INDEPENDIENTE de la del bib historico**: `paper/micai/refs.bib` sigue inmutable y ligado al PDF retirado. Se llama «candidates» porque sin manuscrito nuevo no hay citas nuevas; el `refs.bib` final se generara solo con las claves citadas, y las 26 citas del manuscrito retirado NO se atribuyen a este contexto. Autores sin truncar, identificador localizable en las 57. |
| Manuscrito nuevo: figura del soporte por clase, espanol | `reports/micai2027/figuras/soporte-es.svg` | `2233b75c3a70ba939c67b518c3f29ad3` | 69376 | `a62b504` | SELLADO | Primera figura del manuscrito NUEVO. Ancho fisico exacto de `\textwidth` de LNCS (4,8031 in), sin escala de colocacion; tipografia impresa por encima de 8 pt; clases de cola distinguidas por color **y** trama. El banco se rotula PASTIS, no PASTIS-R. Reproducible byte a byte: dos generaciones seguidas dan el mismo MD5. Sus dos insumos son SELLADO, asi que es citable. |
| Manuscrito nuevo: figura del soporte por clase, ingles | `reports/micai2027/figuras/soporte-en.svg` | `b09217baf4880181706169030675ae37` | 68501 | `a62b504` | SELLADO | La misma figura con los rotulos en ingles, que es la lengua del envio. Mismos insumos, mismo contrato fisico, misma reproducibilidad. El `.pdf` de cada una es la misma figura en otro formato y no lleva fila propia: el ledger sella un representante por figura. |
| Fase 6: bibliografia derivada de la matriz verificada | `paper/micai/refs.bib` | `1452703945c24b628ba6adba2c9d7a86` | 20165 | `b072850` | SELLADO | 54 entradas. Se generan desde la matriz resuelta por API mas un fichero de correcciones verificadas a mano, cada una con su motivo: una de ellas corregia un DOI que resolvia a un dataset distinto del articulo. |
| Fase 6: PDF de envio, version anonima | `paper/micai/main.pdf` | `85261661509cbf7f4deea318b890c3d7` | 633726 | sin seguimiento en git (`*.pdf` global) | SELLADO | 15 paginas A4, cero errores, cero overfull, gate de identidad en verde. `make micai-pdf` lo reconstruye byte a byte desde las fuentes versionadas, asi que el sello vale aunque el binario no viaje. |
| Fase 6: figura de parcelas reales del fold retenido | `reports/paper_micai/figuras/parcelas.svg` | `927a17710e3afc6db31291ca360398c0` | 310658 | `5cb0831` | SELLADO | Tres parches en color natural y su anotacion agronomica; ancla el articulo en el dato. |
| Fase 6: figura del soporte por clase de los dos bancos | `reports/paper_micai/figuras/soporte.svg` | `1c5fc41811a3bf307a82f36931f99d70` | 70411 | `8ab77bd` | SELLADO | Escala logaritmica; la cola del segundo banco es degenerada, con dos y siete parcelas. **Regenerada**: rotulaba el banco como PASTIS-R y el articulo usa el optico. Sus insumos siguen SELLADO, asi que la figura sigue siendo citable. |
| Fase 6: figura de composicion de leyendas | `reports/paper_micai/figuras/leyendas.svg` | `1df574718eb3408f90e48152fbc3ef53` | 60617 | `5cb0831` | OBSOLETO | Ensena que la retirada por soporte suelta la colza y conserva las dos praderas. **Hereda el estado de sus insumos**: sale de `fase4/replica_por_bloque.csv` y `fase4/replica_contrastes.json`, los dos OBSOLETO. Estaba marcada SELLADO mientras `fase4/replica.svg` -el mismo caso- si estaba marcada; el estado se ponia a mano y salio distinto para casos iguales. Ahora lo comprueba una prueba. |
| Fase 6: figura de la frontera en cobertura | `reports/paper_micai/figuras/cobertura.svg` | `fa1e1e7b66ad35b439b7dbabf26c53c5` | 65871 | `5cb0831` | OBSOLETO | La misma medicion contra cobertura entregada, como la lee la literatura selectiva. **Hereda el estado de sus insumos**: `fase3/frontera_por_bloque.csv`, `fase3/frontera_contrastes.json` y `fase4/replica_por_bloque.csv`, los tres OBSOLETO. |
| Fase 6: manuscrito en espanol, fuente | `paper/micai/main_es.tex` | `09382042222bd8b1f81f7d0c939dc039` | 4547 | `6fa433d` | SELLADO | Version de lectura y revision del equipo; la de envio es main.tex. |
| Fase 6: manuscrito en espanol, PDF | `paper/micai/main_es.pdf` | `9f1acce383627559d01b1b35aa6f8b32` | 641991 | sin seguimiento en git (`*.pdf` global) | SELLADO | 15 paginas A4, cero errores, cero overfull, gate de identidad en verde. |
| Fase 8: camera-ready en ingles | `paper/micai/main_cr.pdf` | `d78d76123770c4a3dde0f95a74c9640b` | 647319 | sin seguimiento en git (`*.pdf` global) | SELLADO | Autor de correspondencia Arthur Jafed Zizumbo-Velasco, los dos apellidos con guion, ORCID, afiliacion, agradecimientos sin atribuir autoria del codigo y declaracion de que no hubo financiamiento. Sale del MISMO main.tex; el cuerpo es identico al anonimo. |
| Fase 8: camera-ready en espanol | `paper/micai/main_cr_es.pdf` | `28f56176cc1c34152e369671d489e19b` | 649918 | sin seguimiento en git (`*.pdf` global) | SELLADO | Version de lectura del equipo, no de envio. Mismo cuerpo que su anonima. |
| Registro de la búsqueda sistemática de la fase 0 | `reports/paper_micai/fase0/search_log.csv` | `ac14cc4e5db38c7976c1c6b6c4af05e1` | 12831 | `4052b0b` | SELLADO | Consulta, fuente, fecha, código HTTP y registros. |
| Registro de las consultas manuales de buscador | `reports/paper_micai/fase0/search_log_manual.csv` | `3de611cd854356095debdb34fb0cbbc5` | 2163 | `4052b0b` | SELLADO | Las seis consultas tipo Google Scholar. |
| Candidatos devueltos por la búsqueda | `reports/paper_micai/fase0/search_candidates.csv` | `c56c4aad185e7a08a4dc7a383eaef35f` | 162047 | `4052b0b` | SELLADO | Sin filtrar, tal como los devolvió cada API. |
| Correcciones bibliográficas verificadas contra fuente primaria | `reports/paper_micai/fase0/related_work_overrides.csv` | `2dc4f60f132757ce07b9847b05397986` | 14681 | `cfc07f2` | SELLADO | Cada corrección con su motivo escrito. El generador del `.bib` las superpone sobre lo que devolvió la API, para que la resolución automática conserve su procedencia y la corrección no se escriba a mano en un fichero generado. Aquí vive, entre otras, que `garnot2021pastis` introduce PASTIS óptico y NO PASTIS-R. |
| Matriz de trabajo relacionado redactada | `reports/paper_micai/fase0/related_work_matrix.csv` | `4f546d4871198014b4437e2ee70df809` | 26225 | `a35e96f` | SELLADO | Método, fortaleza, límite y hueco por entrada. |
| Matriz de trabajo relacionado verificada por API | `reports/paper_micai/fase0/related_work_verified.csv` | `633b122070d9b03248132487989a993d` | 37039 | `a35e96f` | SELLADO | 44 entradas, 44 en estado OK. |

## Cifras sin artefacto

Estas filas no tienen archivo y por tanto **ninguna cifra que dependa de ellas puede
imprimirse**. Se conservan aquí para que la ausencia sea explícita y auditable.

| Elemento del paper | Artefacto esperado | MD5 | Bytes | Commit | Estado | Nota |
|---|---|---|---|---|---|---|
| Métricas de uso de herramientas del copiloto | `reports/agent_bench/us049_system_eval.json` | - | - | - | SIN_ARTEFACTO | Solo existe en la VM H100. Sin él, la tabla de herramientas sale del texto. Depende de Arthur. |
| Embeddings AlphaEarth de Italia 2018 | `data/features/alphaearth_italia_2018.parquet` | - | - | - | SIN_ARTEFACTO | El puntero .dvc está en git desde julio pero el archivo nunca subió al bucket. Depende de Arthur. |
| Informe DE4 (Baja Sajonia) | `checkpoints/transfer/voting-italia/de4_2023/report.json` | - | - | - | SIN_ARTEFACTO | Unas 35 cifras del manuscrito dependen de este archivo. Depende de Arthur. |
| Delta AlphaEarth frente a Sentinel-2 en EuroCropsML | `data/transfer/eurocropsml_alphaearth_vs_s2_delta.parquet` | - | - | - | SIN_ARTEFACTO | Sostiene la cifra +0,111 F1 en k=1. Sin artefacto: la cifra sale del texto. |
| Few-shot de WorldCereal, Brasil | `data/transfer/worldcereal_fewshot_results.parquet` | - | - | - | SIN_ARTEFACTO | Regenerable en CPU con `scripts/build_worldcereal_tropical_figure.py` desde el parquet crudo sellado; hoy fuera del cuerpo. |
| Few-shot de WorldCereal, India | `data/transfer/worldcereal_fewshot_india.parquet` | - | - | - | SIN_ARTEFACTO | Regenerable en CPU con el mismo script desde el parquet crudo sellado; hoy fuera del cuerpo. |
| Transferencia PASTIS a BreizhCrops (Bretaña) | `data/transfer/pastis_to_breizhcrops.parquet` | - | - | - | SIN_ARTEFACTO | Sostiene la cifra F1 0,21 de Bretaña. Sin artefacto: la cifra sale del texto. |
| Registros de la ablación de bandas de FarSLIP | `reports/farslip/logs/` | - | - | - | SIN_ARTEFACTO | Sostiene la tabla de ablación de bandas. Sin artefacto: la tabla sale del texto. |

## Reproducibilidad: qué artefacto tiene productor en el repositorio

Sellar un archivo garantiza que no cambió; no garantiza que se pueda volver a
generar. Estos son los productores localizados con `grep` sobre el código, no de
memoria.

| Artefacto | Productor en el repositorio |
|---|---|
| `ml/eval/oof/oof_parcel_*_fold5.parquet` y `manifest.json` | `ml/eval/oof/dump_oof.py` |
| `reports/ensemble/us043_farslip_summary.json` | `scripts/run_us043_farslip_ensembles.py` |
| `reports/ensemble/metrics/us043_farslip_stacking_blending.csv` | `scripts/run_us043_farslip_ensembles.py` |
| `reports/ensemble/metrics/ec_neighborhood_result.json` | `ml/ensemble/ec_neighborhood.py` |
| `reports/ensemble/metrics/comparison_us040.csv` | `scripts/run_us040_ensembles.py` |
| `reports/agent_bench/perceiver_champion_eval_v2.json` | `ml/eval/perceiver_champion_eval.py` |
| `reports/segmentation/sen4agrinet_transfer_result.json` | `ml/transfer/sen4agrinet_domain_gap.py` |
| `reports/us079_figs/ablation_compare.json` | `scripts/run_us079_ablation_analysis.py` |
| `data/transfer/worldcereal_*.parquet` | `ml/transfer/worldcereal_tropical.py` |
| `data/transfer/multiregion_*.parquet` | `ml/transfer/multiregion_model.py` |
| `reports/paper_micai/fase0/search_*.csv` | `scripts/paper_micai_lit_search.py` |
| `reports/paper_micai/fase0/related_work_verified.csv` | `scripts/paper_micai_ref_verify.py` |
| `reports/paper_micai/fase1/parcel_gt_fold5*.{parquet,csv,json}` | `scripts/paper_micai_seal_fold5.py` |
| `reports/paper_micai/fase2/*` | `scripts/run_paper_micai_fase2.py` sobre `ml/eval/paper_micai_arbitration.py` |
| `reports/paper_micai/fase3/*` | `scripts/run_paper_micai_fase3.py` sobre `ml/eval/paper_micai_coverage.py`, figura con `scripts/build_paper_micai_fase3_figure.py` |
| `reports/paper_micai/fase4/breizhcrops_*` | `scripts/build_breizhcrops_features.py` sobre `ml/ingest/breizhcrops_loader.py` y `ml/features/breizhcrops_features.py` |
| `reports/paper_micai/diagnostico/*` | `scripts/run_paper_micai_diagnostico_protocolo.py` |
| `reports/paper_micai/equidad/*` | `scripts/run_paper_micai_equidad.py` |
| `reports/paper_micai/potencia/*` | `scripts/run_paper_micai_potencia_disparidad.py` |
| `reports/paper_micai/bloques/*` | `scripts/run_paper_micai_bloques.py` |
| `reports/paper_micai/prereg/*` | `scripts/run_paper_micai_parametros_prereg.py` |
| `reports/us172/*.pdf` | `make us172-adjuntos` sobre `docs/paper/perdidas-protocolo.md` |
| `reports/paper_micai/fase4/replica_*` | `scripts/run_paper_micai_fase4.py` sobre el mismo `ml/eval/paper_micai_coverage.py` de la fase 3, figura con `scripts/build_paper_micai_fase4_figure.py` |
| `paper/micai/refs.bib` | `scripts/build_paper_micai_bib.py` sobre `related_work_verified.csv` mas `related_work_overrides.csv` |
| `paper/micai/main.pdf` | `make micai-pdf`, verificado por `make micai-anon-check` |
| `paper/micai/main_es.pdf` | `make micai-pdf-es`, verificado por `make micai-anon-check` |
| `paper/micai/main_cr*.pdf` | `make micai-pdf-cr`, que ademas exige que el gate de anonimato FALLE sobre ellos |
| `reports/paper_micai/figuras/parcelas.*` | `scripts/build_paper_micai_patch_figure.py` sobre `data/PASTIS-R` |
| `reports/paper_micai/figuras/{soporte,leyendas,cobertura}.*` | `scripts/build_paper_micai_extra_figures.py` |

**Con cálculo versionado pero sin driver.** En estos seis, la lógica sí está en el
repositorio; lo que no está versionado es el guion que la invocó y escribió el archivo.
Regenerarlos no exige reescribir el cálculo, sino volver a escribir su punto de entrada.

| Artefacto | Qué cifra sostiene | Dónde está el cálculo |
|---|---|---|
| `reports/ensemble/metrics/us043_farslip_grid.csv` | 0,7486 y 0,8495 del Stacking-5, y 0,7470 del Stacking-3 | `ml/ensemble/stacking.py` y `blending.py`, más `_stacking_metrics` y `_blending_metrics` de `scripts/run_us043_farslip_ensembles.py`, que hoy solo barre un universo |
| `reports/ensemble/metrics/us043_honest_dropout_curve.csv` | toda la curva calidad-cobertura, que es la contribución central | `ml.eval.per_class_analysis.honest_class_dropout_curve`, con el protocolo libre de fuga ya implementado |
| `reports/ensemble/metrics/us043_winner_cardinality_curve.csv` | la curva por soporte acumulado | `ml.eval.per_class_analysis.cardinality_cutoff_curve` |
| `reports/ensemble/metrics/us043_winner_per_class.csv` | el desempeño por clase del campeón | `ml.eval.per_class_analysis.per_class_report` |
| `reports/farslip/metrics/us037_farslip_fiel_vs_alphaearth.csv` | la comparación FarSLIP frente a AlphaEarth | `ml/eval/embedding_separability.py` |
| `data/transfer/eurocropsml_fewshot_results.parquet` | la curva k-shot LV a EE | `ml.transfer.eurocropsml_alphaearth_fewshot.train_xgb_kshot_alphaearth` |

Consecuencia para la fase 2: la curva calidad-cobertura **se vuelve a generar llamando a
`honest_class_dropout_curve` desde las posteriores OOF selladas**, no se hereda del CSV.
El CSV queda como comprobación cruzada del resultado nuevo.

## Comprobación cruzada ya hecha

El eje de cobertura de la curva **ya reproduce**. Recalculando el soporte de cada
conjunto de clases retenidas desde `reports/paper_micai/fase1/parcel_gt_fold5.parquet`,
derivado de PASTIS-R de forma independiente, los siete valores de
`n_parcels_fold5` coinciden exactamente con los del CSV sellado: 16 640, 16 092, 15 783,
14 925, 14 200, 13 624 y 13 311. Lo que queda por reproducir es la columna de F1-macro.

## Cómo se usa

```bash
make paper-artifacts-check     # recalcula los MD5 y compara con esta tabla
```

Al regenerar un artefacto hay que actualizar su fila en la misma confirmación que lo
cambia. Un MD5 que no coincide no es un falló del gate: es una cifra del artículo que
acaba de dejar de ser reproducible.

## Verificación del propio gate

El gate se probó en negativo antes de confiar en él: se añadió un salto de línea a
`reports/paper_micai/fase0/search_log_manual.csv` y `make paper-artifacts-check` falló
con el MD5 nuevo frente al sellado; al restaurar el archivo volvió a verde. Un control
que nunca se ha visto fallar no sirve de control.

Además, los once parquets OOF sellados aquí coinciden con el `md5` que registra su
propio archivo `.dvc`, de modo que un clon limpio con `dvc pull` obtiene exactamente
los mismos bytes que produjeron las cifras.

Probado sobre un clon limpio del repositorio: **23 de los 37 artefactos sellados
verifican solo con `git clone`**, entre ellos todo el sello del fold 5 y toda la
evidencia de la fase 0. Los otros 14 son punteros de DVC y el gate los separa con un
mensaje accionable (`ejecuta dvc pull`) en lugar de darlos por sello roto.
