## Results

Evaluated on a held-out test set of 200 examples from `NousResearch/hermes-function-calling-v1`.

| Metric             | Base Qwen2.5-1.5B | ToolForge (QLoRA) | Î”        |
|--------------------|:-----------------:|:-----------------:|:--------:|
| Format Accuracy    | 90.0%            | 99.0%            | +9.0%    |
| Function Accuracy  | 65.5%            | 72.5%            | +7.0%    |
| Argument F1        | 0.689           | 0.734           | +0.045  |
| Hallucination Rate | 0.0%            | 0.5%            | +0.5%    |

> Hallucination rate: lower is better. All other metrics: higher is better.
