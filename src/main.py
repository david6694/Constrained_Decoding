import argparse
import json
import math
import os
import re
import traceback
from typing import Any


from llm_sdk import Small_LLM_Model


# Shared regex patterns used by the prompt parsers below.
NUMBER_PATTERN = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
QUOTED_PATTERN = r"'([^']*)'|\"([^\"]*)\""


def main() -> None:
    base_dir = os.path.dirname(__file__)
    default_funcs = os.path.join(base_dir, "data", "input",
                                 "functions_definition.json")
    default_input = os.path.join(base_dir, "data", "input",
                                 "function_calling_tests.json")
    default_output = os.path.join(
        os.path.abspath(os.path.join(base_dir, "..", "data", "output")),
        "function_calling_results.json",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default=default_funcs,
        help="Path to the tool definition JSON file.",
    )
    parser.add_argument(
        "--input",
        default=default_input,
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    def _find_input_file(filename: str) -> str:
        # Prefer package data first, then the repo-level data directory.
        candidate = os.path.join(base_dir, "data", "input", filename)
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.abspath(
            os.path.join(base_dir, "..", "data", "input", filename)
        )
        if os.path.exists(candidate):
            return candidate
        # Fall back to the plain filename so callers can resolve it from CWD.
        return filename

    # LOAD TESTS
    try:
        # Resolve tests path: prefer explicit existing path.
        if args.input and os.path.exists(args.input):
            tests_path = args.input
        elif args.input:
            # A path was provided but doesn't exist; try resolving by
            # basename within package/data locations
            tests_basename = os.path.basename(args.input)
            tests_path = _find_input_file(tests_basename)
        else:
            tests_path = _find_input_file("function_calling_tests.json")
        with open(tests_path, "r", encoding="utf-8") as f:
            test_prompts = json.load(f)

        # LOAD TOOLS
        if args.functions_definition and os.path.exists(
            args.functions_definition
        ):
            functions_path = args.functions_definition
        elif args.functions_definition:
            funcs_basename = os.path.basename(args.functions_definition)
            functions_path = _find_input_file(funcs_basename)
        else:
            functions_path = _find_input_file("functions_definition.json")
        with open(functions_path, "r", encoding="utf-8") as f:
            tools_json = f.read()
        functions = json.loads(tools_json)
    except FileNotFoundError:
        print("Error: missing input file.")
        return
    except json.JSONDecodeError:
        print("Error: malformed JSON input file.")
        return

    # MODEL LOADING
    print("⚙️ Loading the model, please wait...\n")
    model = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
    print("\n✅ Done\n")

    def logsumexp(xs: list[float]) -> float:
        m = max(xs)
        s = sum(math.exp(x - m) for x in xs)
        return m + math.log(s)

    def extract_quoted_parts(prompt_text: str) -> list[str]:
        # Keep only the actual quoted payload from the regex matches.
        quoted_parts = re.findall(QUOTED_PATTERN, prompt_text)
        return [a or b for (a, b) in quoted_parts]

    def score_token_sequence(
        prefix_ids: list[int], candidate_ids: list[int]
    ) -> float:
        # Score the candidate as the log-probability after the prefix.
        total = 0.0
        cur = list(prefix_ids)
        for tok in candidate_ids:
            logits = model.get_logits_from_input_ids(cur)
            lse = logsumexp(logits)
            total += (logits[tok] - lse)
            cur.append(tok)
        return total

    def extract_params_from_prompt(
        prompt_text: str, param_schema: dict[str, Any]
    ) -> dict[str, Any]:
        # Fill each parameter slot with the best literal match we can find.
        nums = re.findall(NUMBER_PATTERN, prompt_text)
        nums = [float(x) for x in nums]
        quoted = extract_quoted_parts(prompt_text)

        params: dict[str, Any] = {}
        num_idx = 0
        str_idx = 0
        for key, spec in param_schema.items():
            ptype = spec.get("type", "string")
            if ptype == "number":
                if num_idx < len(nums):
                    params[key] = float(nums[num_idx])
                    num_idx += 1
                else:
                    params[key] = 0.0
            else:  # assume string
                if str_idx < len(quoted):
                    params[key] = quoted[str_idx]
                    str_idx += 1
                else:
                    params[key] = prompt_text
        return params

    def normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def score_function_by_rules(
        prompt_text: str, function_def: dict[str, Any]
    ) -> float:
        # Coarse keyword scoring to narrow the search before model scoring.
        prompt = normalize_text(prompt_text)
        name = function_def["name"]
        description = normalize_text(function_def.get("description", ""))
        haystack = f"{name} {description}"

        score = 0.0

        def has_any(words: list[str]) -> bool:
            return any(word in prompt for word in words)

        if name == "fn_add_numbers":
            if has_any(["sum", "add", "plus", "total", "combined", "numbers"]):
                score += 8.0
            if has_any([
                "replace",
                "reverse",
                "greet",
                "square root",
                "sqrt",
                "substitute",
            ]):
                score -= 10.0

        elif name == "fn_greet":
            if has_any(["greet", "hello", "hi", "hey", "name"]):
                score += 8.0
            if has_any([
                "reverse",
                "square root",
                "replace",
                "substitute",
                "sum",
                "add",
            ]):
                score -= 10.0

        elif name == "fn_reverse_string":
            if has_any(["reverse", "backwards", "flip"]):
                score += 8.0
            if has_any(["string", "text", "word"]):
                score += 1.5
            if has_any([
                "replace",
                "square root",
                "sum",
                "greet",
                "substitute",
            ]):
                score -= 10.0

        elif name == "fn_get_square_root":
            if has_any(["square root", "sqrt", "root"]):
                score += 9.0
            if has_any([
                "replace",
                "reverse",
                "greet",
                "substitute",
                "sum",
                "add",
            ]):
                score -= 10.0

        elif name == "fn_substitute_string_with_regex":
            if has_any([
                "replace",
                "substitute",
                "swap",
                "change",
                "all occurrences",
            ]):
                score += 9.0
            if has_any([
                "numbers",
                "digits",
                "vowels",
                "letters",
                "word",
                "regex",
                "pattern",
            ]):
                score += 4.0
            if has_any(["sum", "add", "square root", "greet", "reverse"]):
                score -= 10.0

        for token in set(description.split()) & set(prompt.split()):
            if len(token) > 3:
                score += 0.25

        for token in set(haystack.split()) & set(prompt.split()):
            if len(token) > 3:
                score += 0.15

        return score

    def infer_regex_and_replacement(
        prompt_text: str,
    ) -> tuple[str | None, str | None]:
        # Infer a regex and replacement from the wording of the prompt.
        prompt = prompt_text.lower()

        regex = None
        if any(word in prompt for word in ["vowel", "vowels"]):
            regex = r"[aeiouAEIOU]"
        elif any(
            word in prompt for word in ["number", "numbers", "digit", "digits"]
        ):
            regex = r"\d+"
        elif "word" in prompt:
            quoted_words = extract_quoted_parts(prompt_text)
            if quoted_words:
                regex = rf"\b{re.escape(quoted_words[0])}\b"
            else:
                regex = r"\w+"

        replacement = None
        if any(
            word in prompt
            for word in [
                "asterisk",
                "asterisks",
                "star",
                "stars",
            ]
        ):
            replacement = "*"
        else:
            after_with = re.split(r"\bwith\b", prompt_text, maxsplit=1)
            if len(after_with) == 2:
                tail = after_with[1].strip()
                quoted_tail = extract_quoted_parts(tail)
                if quoted_tail:
                    replacement = quoted_tail[0]
                else:
                    replacement = tail.split()[0] if tail else ""

        return regex, replacement

    # MAIN LOOP
    results = []
    func_map = {f["name"]: f for f in functions}
    func_token_ids = {}
    for fname in func_map:
        ids = model.encode(fname).tolist()[0]
        func_token_ids[fname] = ids

    for test in test_prompts:
        user_prompt = test["prompt"]
        print(f"Prompt: {user_prompt}")
        try:
            # Rank tools first with rules, then refine the top candidates.
            scored_functions = []
            for function_def in functions:
                rule_score = score_function_by_rules(user_prompt, function_def)
                scored_functions.append(
                    (rule_score, function_def["name"], function_def)
                )

            scored_functions.sort(
                key=lambda item: (item[0], len(item[1])),
                reverse=True,
            )
            if not scored_functions or scored_functions[0][0] <= 0:
                raise ValueError("No matching function")

            score_by_name = {
                name: score for score, name, _ in scored_functions
            }

            best_name = None
            best_score = -1e300
            top_candidates = scored_functions[:2]

            try:
                short_prefix = (
                    "You have access to these tools:\n"
                    + tools_json
                    + "\n\n"
                    + user_prompt
                    + "\nPick the best function name from the list above "
                    + "and output only the function name (no JSON):\n"
                )
                prefix_ids = model.encode(short_prefix).tolist()[0]

                for _, fname, _ in top_candidates:
                    fid_seq = func_token_ids[fname]
                    try:
                        sc = score_token_sequence(prefix_ids, fid_seq)
                    except Exception:
                        sc = -1e300
                    sc += score_by_name[fname]
                    if sc > best_score:
                        best_score = sc
                        best_name = fname
            except Exception:
                best_name = None

            if best_name is None:
                best_name = scored_functions[0][1]

            # Extract parameters after the function choice is finalized.
            param_schema = func_map[best_name].get("parameters", {})
            parameters = extract_params_from_prompt(user_prompt, param_schema)

            if best_name == "fn_substitute_string_with_regex":
                regex, replacement = infer_regex_and_replacement(user_prompt)
                if "source_string" in param_schema:
                    in_match = re.search(
                        r"\bin\b\s+(?:'([^']*)'|\"([^\"]*)\")",
                        user_prompt,
                        flags=re.IGNORECASE,
                    )
                    if in_match:
                        parameters["source_string"] = (
                            in_match.group(1) or in_match.group(2)
                        )
                    else:
                        quoted_parts = extract_quoted_parts(user_prompt)
                        parameters["source_string"] = (
                            quoted_parts[-1] if quoted_parts else user_prompt
                        )
                if regex is not None and "regex" in param_schema:
                    parameters["regex"] = regex
                if replacement is not None and "replacement" in param_schema:
                    parameters["replacement"] = replacement

            elif best_name == "fn_reverse_string":
                quoted_parts = extract_quoted_parts(user_prompt)
                if quoted_parts and "s" in param_schema:
                    parameters["s"] = quoted_parts[0]

            elif best_name == "fn_greet":
                m = re.search(
                    r"\bgreet\b\s+(.+)$",
                    user_prompt,
                    flags=re.IGNORECASE,
                )
                if m and "name" in param_schema:
                    parameters["name"] = m.group(1).strip().strip("'\"")

            elif best_name == "fn_get_square_root":
                if "a" in param_schema:
                    nums = re.findall(NUMBER_PATTERN, user_prompt)
                    if nums:
                        parameters["a"] = float(nums[0])

            result_obj = {
                "prompt": user_prompt,
                "name": best_name,
                "parameters": parameters,
            }
            results.append(result_obj)

            print(
                f"  Chosen function: {best_name}, parameters: {parameters}"
            )
        except ValueError:
            print("Error: no function matching this prompt.")
            continue
        except Exception:
            print("Error: unexpected failure while processing this prompt.")
            traceback.print_exc()
            continue

    # OUTPUT JSON
    out_path = (
        args.output
        if args.output is not None
        else os.path.join(
            os.path.abspath(os.path.join(base_dir, "..", "data", "output")),
            "function_calling_results.json",
        )
    )
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(results)} calls to {out_path}")


if __name__ == "__main__":
    main()
