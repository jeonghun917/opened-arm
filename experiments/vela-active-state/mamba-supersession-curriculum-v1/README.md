# VELA Mamba supersession curriculum v1

Purpose: test whether the native long-history overwrite/supersession failure is mainly a curriculum problem rather than simple undertraining.

Three budget-matched arms start from the same Mamba-130M base and use the same x_proj-only update recipe for 10 epochs / 160 optimizer steps (batch size 1):

1. `baseline`: original 16 single-correction examples.
2. `mixed`: 8 original correction examples + 8 three-stage supersession examples.
3. `supersession_only`: 16 three-stage supersession examples.

All arms are evaluated on the unchanged 38-fixture native long-history suite plus the original held-out task. No selector, migration, checkpoint replay, or target-state injection is used.

Primary diagnostic: whether the `superseded` family rises above the baseline 0/13 while ordinary held-out capability remains measurable.
