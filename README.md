# Constrained_Decoding

## Description
This project implements a lightweight function-calling pipeline. Given a user prompt and a list of available tool definitions, the program selects the most relevant function and fills its parameters with values extracted from the prompt.

The goal is to transform natural language requests into structured tool calls and write the final results to a JSON file. The project combines rule-based heuristics with a small language model to improve tool selection and parameter extraction.

## Instructions
### Installation
The project is designed to run with Python, `uv`, and `make`.

```bash
make install
```

This creates a virtual environment with `uv` and syncs the dependencies declared in `pyproject.toml`, including the local `llm_sdk` package in editable mode.

### Execution
```bash
uv run python -m src [--functions_definition <function_definition_file>] [--input <input_file>] [--output <output_file>]
```

By default, the program reads the function definitions and input files from `data/input/` and writes the output to `data/output/`.

Example:

```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json
```

If you prefer the Makefile shortcut, `make run` still invokes the default `uv run python -m src` command.

### Optional checks
```bash
make lint
```

## Algorithm Explanation
The decoding strategy is constrained by the finite tool list provided in `functions_definition.json`.

1. The program first scores each available function with keyword rules built from the prompt.
2. It keeps the strongest candidates and then refines the ranking with the language model by estimating the log-probability of each function name after a prompt-aware prefix.
3. Once the best function is chosen, the program extracts parameters from the prompt using simple parsers for numbers and quoted strings.
4. For some tools, extra heuristics are applied, such as regex inference for substitution requests or dedicated extraction for names and square-root inputs.

This approach constrains selection to the known function set (rule-based filtering + log-probability reranking over the allowed tool names) rather than performing full token-level grammar/FSM logit masking during free-form JSON generation. The model never invents arbitrary tool names, keeping outputs compatible with the expected JSON schema.

## Design Decisions
The implementation favors a small and readable pipeline over a fully generative solution.

- Rule-based scoring is used first because it is fast and provides a strong prior for obvious prompts.
- Model scoring is used second to disambiguate close candidates without evaluating every possible token sequence.
- Parameter extraction is intentionally simple, using deterministic parsing for numbers, quoted strings, and a few special cases.
- The output format is kept as JSON so it can be consumed by automated graders or downstream tooling.

## Performance Analysis
Accuracy is improved by combining heuristics with model scoring. The rule layer handles clear prompts quickly, while the model layer helps when wording is less direct. This hybrid design is more reliable than using only keyword matching, especially for prompts that contain similar vocabulary across multiple tools.

Speed is acceptable for a small function set. The expensive step is model scoring, but it is limited to the top candidates instead of every tool. Reliability is good as long as the input files are present and the prompt language stays close to the supported patterns. When a file is missing, the program now exits cleanly with a short error message.

## Challenges Faced
The main difficulty was balancing simplicity, correctness, and robustness.

- Distinguishing between functions with overlapping vocabulary required a layered ranking strategy.
- Extracting parameters from free-form text was tricky because prompts do not always mention values in a consistent format.
- Regex substitution requests needed extra logic because the replacement value is often implied rather than explicit.
- Missing input files had to be handled cleanly so the program does not fail with an unclear traceback.

These issues were addressed with fallback rules, special-case parsing, and a small `try/except` around the input loading step.

## Testing Strategy
Validation was done with the provided function-calling test prompts in `data/input/`.

- The output JSON was checked to confirm that the chosen function name belongs to the known tool list.
- Parameter extraction was tested on prompts containing numbers, quoted strings, greetings, reverse-string requests, square-root requests, and regex substitution requests.
- The file-loading path was tested by handling missing input files gracefully.
- Linting is available through `make lint` to catch style and type issues early.

## Example Usage
```bash
make install
uv run python -m src
```

After execution, inspect the generated file:

```bash
data/output/function_calling_results.json
```

Example result shape:

```json
{
  "prompt": "add 3 and 5",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 3,
    "b": 5
  }
}
```

## Resources
- Python documentation: https://docs.python.org/3/
- JSON module documentation: https://docs.python.org/3/library/json.html
- Regular expression documentation: https://docs.python.org/3/library/re.html
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
- PyTorch documentation: https://pytorch.org/docs/

### AI Usage
AI was used to help draft this README structure and to rewrite the project behavior into clear technical English. It was also used to summarize the algorithm, testing approach, and limitations based on the existing implementation in `src/main.py`.