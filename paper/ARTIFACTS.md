# ARTIFACTS — ledger de custodia del manuscrito MICAI 2027

Cada cifra impresa en el artículo debe re-derivarse desde un archivo de esta tabla.
Lo que no aparece aquí no se imprime. El gate `make paper-artifacts-check` recalcula
el MD5 de cada fila sellada y falla si algo cambió sin registrarse.

**Sellado el**: 2026-09-03 · **HEAD del repositorio**: `a8386d8`
**Fase**: 6 de [`docs/plan-micai-2027.md`](../docs/plan-micai-2027.md) ·
**Ángulo vigente**: [`ADR-013`](../docs/decisions/ADR-013-angulo-micai.md) (reencuadre).

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
| Miembros y protocolo: manifiesto del arnés OOF (fold 5, 18 clases, code_version) | `ml/eval/oof/manifest.json` | `1737fd6b38ae6cec6cec669640bca849` | 3213 | `23cb470` | SELLADO | Declara fold 5 held-out, 18 clases y el commit que produjo las posteriores. |
| Universo del ensamble: posteriores por parcela, U-Net | `ml/eval/oof/oof_parcel_unet_fold5.parquet` | `ff36268a0d9bdbaa732374fa031fcf3f` | 2015608 | `23cb470` (.dvc) | SELLADO | Miembro denso espacial. |
| Universo del ensamble: posteriores por parcela, DeepLabv3+ | `ml/eval/oof/oof_parcel_deeplabv3plus_fold5.parquet` | `6a2b13aa4213b73349310f767934635c` | 2075019 | `23cb470` (.dvc) | SELLADO | Miembro denso espacial. |
| Universo del ensamble: posteriores por parcela, SegFormer | `ml/eval/oof/oof_parcel_segformer_fold5.parquet` | `99d8b9c0e766d1a42f1cb6543bb7fba6` | 2062477 | `23cb470` (.dvc) | SELLADO | Miembro denso espacial. |
| Universo del ensamble: posteriores por parcela, U-TAE | `ml/eval/oof/oof_parcel_utae_fold5.parquet` | `597d867f32f9f7c52ff75702e2cf1c14` | 2058667 | `23cb470` (.dvc) | SELLADO | Miembro temporal; entra en el Stacking-5. |
| Universo del ensamble: posteriores por parcela, TSViT-pheno | `ml/eval/oof/oof_parcel_tsvit-pheno_fold5.parquet` | `b87f9b63ccecf4e5395bb22a307d67e8` | 2071454 | `23cb470` (.dvc) | SELLADO | Miembro del universo campeón. |
| Universo del ensamble: posteriores por parcela, TSViT-pheno-fullm | `ml/eval/oof/oof_parcel_tsvit-pheno-fullm_fold5.parquet` | `16758cb30ec2fc818bf0bed56e4f3a1a` | 2060956 | `fddcb38` (.dvc) | SELLADO | Miembro del segundo universo; es el mejor individual y da el peor stacking. |
| Universo del ensamble: posteriores por parcela, AnySat | `ml/eval/oof/oof_parcel_anysat_fold5.parquet` | `3366dc2469ab82575e82f2d8bfe0ffcc` | 2036192 | `23cb470` (.dvc) | SELLADO | Miembro de modelo de fundación. |
| Universo del ensamble: posteriores por parcela, XGBoost sobre AlphaEarth | `ml/eval/oof/oof_parcel_xgb-alphaearth_fold5.parquet` | `97a083d90b1635dbfc216d837325877f` | 1192508 | `10084d2` (.dvc) | SELLADO | Miembro tabular; fija la intersección de 16 640 parcelas. |
| Universo del ensamble: posteriores por parcela, FarSLIP fine-tuned 18 | `ml/eval/oof/oof_parcel_farslip-ft18_fold5.parquet` | `433325d8ad2cb7c65a26c19065f25295` | 1150480 | `1890bef` (.dvc) | SELLADO | Rama contrastiva del Stacking-5. |
| Universo del ensamble: posteriores por parcela, FarSLIP zero-shot | `ml/eval/oof/oof_parcel_farslip-zeroshot_fold5.parquet` | `28c2b44de3fd13e4cadb3937c5bcbea0` | 1075770 | `1890bef` (.dvc) | SELLADO | Rama contrastiva del Stacking-5. |
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
| Etiquetas por parcela del fold 5, universo compartido por los diez miembros | `reports/paper_micai/fase1/parcel_gt_fold5.parquet` | `dddb1ea9c1b0b15c0f24142e8dd0ec4d` | 68147 | `9eaeb2f` | SELLADO | 16 640 filas derivadas de PASTIS-R. Permite reproducir toda la evaluación sin los 68 GB del dataset. |
| Centroides por parcela del fold 5 | `reports/paper_micai/fase1/parcel_centroids_fold5.parquet` | `5b1ce7b92059e62f640869a51bc8f1e6` | 358870 | `9eaeb2f` | SELLADO | Geometrías necesarias para los sub-folds espaciales del meta-modelo y para el nulo de vecindad. |
| Soporte por clase del fold 5 | `reports/paper_micai/fase1/parcel_gt_fold5_support.csv` | `e9c3b084ad2f8dba43e241f5ff607e46` | 136 | `262a519` | SELLADO | El desbalance del que trata el artículo: de 6 128 parcelas en la clase mayoritaria a 103 en la menor. |
| Procedencia del ground truth del fold 5 | `reports/paper_micai/fase1/parcel_gt_fold5_provenance.json` | `7ad456fbc109edc82d2490683918f65b` | 414 | `262a519` | SELLADO | MD5 del archivo de Zenodo de PASTIS-R, conteos, `code_version` y version de polars. |
| Fase 2: los diez miembros bajo un protocolo único | `reports/paper_micai/fase2/individuales_protocolo_unico.csv` | `59d4c8bfffb0505da27caaf644389e6c` | 626 | `87cccdd` | SELLADO | El mejor individual es tsvit-pheno con F1-macro 0,7367; ninguna combinación libre de fuga lo supera. |
| Fase 2: las cuatro reglas de combinación con su régimen | `reports/paper_micai/fase2/combinaciones.csv` | `036178fa87848b74bc3e5a190b5f96d1` | 483 | `87cccdd` | SELLADO | Separa el refit sobre todo (in-sample) del agrupado espacial (libre de fuga); ahí se ve el origen del 0,7486. |
| Fase 2: bootstrap pareado y McNemar del arbitraje | `reports/paper_micai/fase2/arbitraje_pruebas.json` | `9c2191f7ed5537857b7bdb5b22ae6a11` | 2223 | `87cccdd` | SELLADO | El árbitro no gana al promedio y pierde contra el mejor individual, con intervalos que excluyen el cero. |
| Fase 2: F1 por clase de las cuatro reglas | `reports/paper_micai/fase2/arbitraje_por_clase.csv` | `18493e9b47b5b00d60fe5cb664205d51` | 1556 | `87cccdd` | SELLADO | Muestra que el árbitro retira la clase 10 en silencio: F1 exactamente 0 sobre 355 parcelas. |
| Fase 2: posteriores del árbitro agrupado libre de fuga | `reports/paper_micai/fase2/arbitro_agrupado_posteriores.parquet` | `f0747c3a71108d4fb1c183b40bb2f5a4` | 2325843 | `87cccdd` | SELLADO | Cada parcela predicha por el bloque que no la vio; insumo del nulo de vecindad. |
| Fase 2: cobertura y calidad por bloque y mecanismo | `reports/paper_micai/fase2/cobertura_por_bloque.csv` | `c5e9d5c7f202c764958e243b591ae63b` | 5757 | `8be4f90` | SELLADO | Cada bloque contra UNA leyenda elegida fuera de él, con la leyenda registrada. |
| Fase 2: resumen de la frontera calidad-cobertura | `reports/paper_micai/fase2/cobertura_resumen.csv` | `f2c73a3c79519829f8b4c718bbed2bc1` | 1327 | `8be4f90` | SELLADO | Media, mínimo y máximo entre bloques de los dos mecanismos a igual cobertura. |
| Fase 2: comparativa densa por fold de las variantes de TSViT | `reports/segmentation/metrics/tsvit_pheno_vs_base_fold5.csv` | `44b73e188a1a510e805b062d8d62facc` | 588 | `71ca906` | SELLADO | Da 0,7918 de F1-macro pixel a tsvit-pheno-fullm, frente a 0,2552 de su volcado por parcela: la inconsistencia B2 de la auditoria. |
| Fase 2: contraste de los dos mecanismos con intervalo | `reports/paper_micai/fase2/cobertura_comparacion.json` | `53412f216cfc831e1631f36543733efa` | 4188 | `8be4f90` | SELLADO | Contribución central: retirar clases domina al rechazo por confianza y el IC excluye el cero por debajo de doce clases. |
| Fase 2: barrido completo del refinamiento por vecindad | `reports/paper_micai/fase2/vecindad_barrido.csv` | `4b6399f6c8a9da817e71cee396e6666e` | 1484 | `18e699e` | SELLADO | Ningun punto con alfa mayor que cero mejora al alfa cero sobre el mejor predictor libre de fuga. |
| Fase 2: veredicto del nulo de vecindad con intervalo | `reports/paper_micai/fase2/vecindad_veredicto.json` | `568ffd6739409dda52274b6d0d7c3f14` | 1710 | `18e699e` | SELLADO | Regla R1 de ADR-013: el intervalo incluye el cero, asi que es un nulo acotado. |
| Fase 2: FarSLIP cinco frente a tres miembros | `reports/paper_micai/fase2/farslip_cinco_vs_tres.csv` | `0a83f40f2279cf4c6a19bccaacfad024` | 568 | `18e699e` | SELLADO | Las filas in-sample reproducen 0,7470 y 0,7486; las libres de fuga dan 0,6789 y 0,6794. |
| Fase 2: delta de FarSLIP con intervalo | `reports/paper_micai/fase2/farslip_delta.json` | `b6fe5ac82c3da95cfe339afaa70cdfbc` | 1231 | `18e699e` | SELLADO | +0,0006 con IC de [-0,0024, +0,0034]: el aporte no se distingue de cero fuera del regimen in-sample. |
| Fase 3: la frontera rehecha, por bloque y mecanismo | `reports/paper_micai/fase3/frontera_por_bloque.csv` | `8a10691f779c155067b2e06d60bd0874` | 30960 | sin seguimiento en git | SELLADO | Cuatro series por predictor con estimando alineado y entrega sin oráculo; incluye la leyenda de cada bloque. |
| Fase 3: resumen de la frontera | `reports/paper_micai/fase3/frontera_resumen.csv` | `cf1dce117ca1d6eeacb595596d46b469` | 5808 | sin seguimiento en git | SELLADO | Media, mínimo y máximo entre bloques del F1 alineado, y el F1 nativo al lado, que NO es comparable entre mecanismos. |
| Fase 3: contrastes con intervalo pareado y corrección de Holm | `reports/paper_micai/fase3/frontera_contrastes.json` | `bc282a67b8b69b71e6037c1e9748e0e4` | 15532 | sin seguimiento en git | SELLADO | H1 no se sostiene: a K=9 el IC incluye el cero en los dos predictores y ningún contraste sobrevive a Holm. |
| Fase 3: figura de la frontera | `reports/paper_micai/fase3/frontera.svg` | `d103c5ad7c6a1209f7dab329915e5a97` | 57355 | sin seguimiento en git | SELLADO | Tres mecanismos más el control sin mecanismo, con el criterio principal marcado. |
| Fase 4: caracteristicas por parcela de BreizhCrops (185 columnas, 60 000 parcelas) | `reports/paper_micai/fase4/breizhcrops_features.parquet` | `274a4cc93c79fce8b797e6671cdf1e3d` | 55923894 | `a8386d8` (.dvc) | SELLADO | Submuestreo proporcional de 30 000 por region, semilla 42, mismas 185 caracteristicas que el conjunto primario. |
| Fase 4: soporte por clase y region | `reports/paper_micai/fase4/breizhcrops_soporte.csv` | `026f4d8698cb48862f0108e0a3f9a98b` | 430 | sin seguimiento en git | SELLADO | Siete clases por encima de cien parcelas; girasol con 2 y nueces con 7 en total. |
| Fase 4: procedencia del sellado de caracteristicas | `reports/paper_micai/fase4/breizhcrops_procedencia.json` | `517df9be473300707126e8ebbdc7662d` | 510 | sin seguimiento en git | SELLADO | Semilla, submuestreo, version de codigo y el conteo real de 185 caracteristicas. |
| Fase 4: posteriores dejando una region fuera | `reports/paper_micai/fase4/breizhcrops_posteriores.parquet` | `bcc55122c8047c75f644d852c5880b11` | 2593890 | `a8386d8` (.dvc) | SELLADO | Cada posterior viene de un modelo que no vio la region de esa parcela. |
| Fase 4: replica del protocolo, por bloque y mecanismo | `reports/paper_micai/fase4/replica_por_bloque.csv` | `d85baf968e38869e3437b94283a8c536` | 9955 | sin seguimiento en git | SELLADO | Cuatro series en los dos universos declarados, con la leyenda y la cobertura de cada bloque. |
| Fase 4: contrastes de la replica con Holm | `reports/paper_micai/fase4/replica_contrastes.json` | `d023732d961bfe2d9d04ba10849259fd` | 11802 | sin seguimiento en git | SELLADO | La descomposicion se transporta: 95,1 % denominador en el criterio principal y 100 % donde la cobertura sigue completa. |
| Fase 4: figura de la replica | `reports/paper_micai/fase4/replica.svg` | `3f54d20d9c92505866cef87a2448219d` | 55182 | sin seguimiento en git | SELLADO | Los dos universos lado a lado; las cuatro series coinciden hasta K = 6. |
| Fase 6: manuscrito MICAI, fuente principal | `paper/micai/main.tex` | `9907536ca2a23204b8b616d6b72dd27d` | 3403 | `2d96ca7` | SELLADO | Clase llncs con `\newif\ifanon`; anonimo por defecto, A4 real, `hidelinks`. |
| Fase 6: bibliografia derivada de la matriz verificada | `paper/micai/refs.bib` | `b105ea5913081aa5f8b24f1a332764f6` | 15065 | `2d96ca7` | SELLADO | 44 entradas, todas resueltas por arXiv, Crossref u OpenAlex; generada, no escrita a mano. |
| Fase 6: PDF de envio, version anonima | `paper/micai/main.pdf` | `d3f8039a83d16bdeda7484b7876dbd5a` | 362651 | sin seguimiento en git (`*.pdf` global) | SELLADO | 12 paginas A4, cero errores, cero overfull, gate de identidad en verde. `make micai-pdf` lo reconstruye byte a byte desde las fuentes versionadas, asi que el sello vale aunque el binario no viaje. |
| Registro de la búsqueda sistemática de la fase 0 | `reports/paper_micai/fase0/search_log.csv` | `ac14cc4e5db38c7976c1c6b6c4af05e1` | 12831 | `4052b0b` | SELLADO | Consulta, fuente, fecha, código HTTP y registros. |
| Registro de las consultas manuales de buscador | `reports/paper_micai/fase0/search_log_manual.csv` | `3de611cd854356095debdb34fb0cbbc5` | 2163 | `4052b0b` | SELLADO | Las seis consultas tipo Google Scholar. |
| Candidatos devueltos por la búsqueda | `reports/paper_micai/fase0/search_candidates.csv` | `c56c4aad185e7a08a4dc7a383eaef35f` | 162047 | `4052b0b` | SELLADO | Sin filtrar, tal como los devolvió cada API. |
| Matriz de trabajo relacionado redactada | `reports/paper_micai/fase0/related_work_matrix.csv` | `4f546d4871198014b4437e2ee70df809` | 26225 | `4052b0b` | SELLADO | Método, fortaleza, límite y hueco por entrada. |
| Matriz de trabajo relacionado verificada por API | `reports/paper_micai/fase0/related_work_verified.csv` | `633b122070d9b03248132487989a993d` | 37039 | `4052b0b` | SELLADO | 43 entradas, 43 en estado OK. |

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
| `reports/paper_micai/fase4/replica_*` | `scripts/run_paper_micai_fase4.py` sobre el mismo `ml/eval/paper_micai_coverage.py` de la fase 3, figura con `scripts/build_paper_micai_fase4_figure.py` |
| `paper/micai/refs.bib` | `scripts/build_paper_micai_bib.py` sobre `reports/paper_micai/fase0/related_work_verified.csv` |
| `paper/micai/main.pdf` | `make micai-pdf`, verificado por `make micai-anon-check` |

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

