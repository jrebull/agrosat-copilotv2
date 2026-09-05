# docs/us-work — bitácoras de ejecución (desechables)

Aquí vive la **bitácora** de cada US en vuelo: `us-XXX.md`, escrita por el orquestador en la
Fase 3 y completada por las fases de QA y correcciones. Es estado de ejecución, no contrato:
el contrato es el spec congelado en [`docs/us-planning/`](../us-planning/).

Al cerrar la US (Fase 7) la bitácora se destila a [`docs/us-resolved/us-XXX.md`](../us-resolved/)
y **se borra**. Si una US aparece aquí, está abierta; `make harness-status` lo lista.

Plantilla y ciclo completo: [`docs/orchestration/prompts-optimizers-fable.md`](../orchestration/prompts-optimizers-fable.md).
El formato antiguo (`docs/us-handoff/`, `docs/manual-test/`) es historia: se lee, no se extiende.
