"""Shared helpers for the AI rule-generation pipeline (routes/ai_pipeline.py).

This module holds the prompt-formatting, LLM-call and rule-validation helpers
used by the web pipeline and by services/rule_descriptions.py.

The old interactive command-line workflow that used to live here — main(),
save_rule() and display_rule_preview() — has been removed. The 8-step
RuleWizard and routes/ai_pipeline.py own rule generation now, and the firing
predicate is built there (necessary AND fallback only; helpful/'sufficient' CEs
are never part of the boolean logic).
"""
import json
import re

import litellm

# Reasoning model for deep rule analysis; standard model for lighter calls.
# gpt-5.2 matches the reference pipeline ("deep, structured analysis"; was "o3").
THINKING_MODEL = "gpt-5.2"
VALIDATION_MODEL = "gpt-4.1"


def call_thinking_model(messages, model=THINKING_MODEL):
    """Calls thinking/reasoning model (gpt-5.2)."""
    try:
        print(f"\n[*] Calling {model} for deep reasoning...")
        print("[*] This may take 30-60 seconds as the model thinks through the problem...")

        # Reasoning models (gpt-5.2 / o-series) don't use system messages or temperature.
        # Convert system message to user message if present
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append({"role": "user", "content": msg["content"]})
            else:
                formatted_messages.append(msg)

        response = litellm.completion(
            model=model,
            messages=formatted_messages
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"Error calling thinking model: {e}"


def call_llm(messages, model=VALIDATION_MODEL, temperature=0.7):
    """Calls standard LLM."""
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"Error calling LLM: {e}"


def extract_json_from_response(text):
    """Extracts JSON object from markdown code blocks or raw text."""
    # Try to find JSON in code blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None, "No JSON found in response"

    try:
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON: {e}"


def format_ces_for_prompt(ces_dict):
    """Formats CE dictionary for prompt."""
    lines = []
    for ce_name, ce_data in ces_dict.items():
        lines.append(f"\n**{ce_name}**")
        lines.append(f"Definition: {ce_data['definition']}")
        if 'note' in ce_data:
            lines.append(f"Note: {ce_data['note']}")
        lines.append("Examples:")
        for ex in ce_data.get('examples', [])[:2]:  # Show 2 examples
            lines.append(f"  - Input: \"{ex['input']}\" → {ex['output']}")
    return "\n".join(lines)


def format_rules_for_prompt(rules_dict):
    """Formats rules dictionary for prompt."""
    lines = []
    for rule_name, rule_data in rules_dict.items():
        lines.append(f"\n**{rule_name}**")
        if 'necessary' in rule_data:
            lines.append(f"  Necessary: {rule_data['necessary']}")
        if 'fallback' in rule_data:
            lines.append(f"  Fallback: {rule_data['fallback']}")
        if 'sufficient' in rule_data:
            lines.append(f"  Sufficient: {rule_data['sufficient']}")
    return "\n".join(lines)


def validate_rule(rule_data, ces_dict):
    """Validates rule structure and CE references."""
    issues = []

    # Check required fields
    required_fields = ['rule_name', 'description', 'necessary']
    for field in required_fields:
        if field not in rule_data:
            issues.append(f"Missing required field: {field}")

    # Validate CE references
    all_ces_in_rule = set()
    if 'necessary' in rule_data:
        all_ces_in_rule.update(rule_data['necessary'])
    if 'fallback' in rule_data:
        for group in rule_data['fallback']:
            all_ces_in_rule.update(group)
    if 'sufficient' in rule_data:
        all_ces_in_rule.update(rule_data['sufficient'])

    # Check if CEs exist (either in existing CEs or new CEs)
    available_ces = set(ces_dict.keys())
    if 'new_ces' in rule_data:
        available_ces.update(rule_data['new_ces'].keys())

    unknown_ces = all_ces_in_rule - available_ces
    if unknown_ces:
        issues.append(f"Unknown CEs referenced: {unknown_ces}")

    # Check fallback structure
    if 'fallback' in rule_data:
        if not isinstance(rule_data['fallback'], list):
            issues.append("Fallback must be a list of lists")
        else:
            for i, group in enumerate(rule_data['fallback']):
                if not isinstance(group, list):
                    issues.append(f"Fallback group {i} must be a list")
                if len(group) == 0:
                    issues.append(f"Fallback group {i} is empty")

    return issues
