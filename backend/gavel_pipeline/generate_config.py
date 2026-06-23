import os
import json
import litellm
from datetime import datetime

# --- Configuration ---

CONFIG_GENERATOR_PROMPT = "prompts/config_generator.md"
SCENARIOS_OUTPUT_DIR = "outputs/step_1/generated_scenarios"
CONFIG_GENERATOR_MODEL = "gpt-4.1"  # Model for meta-configuration generation

# --- Helper Functions ---

def load_prompt_template(filepath):
    """Loads a prompt template file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"[CRITICAL] Prompt file not found: {filepath}")
        exit(1)

def call_llm_for_config(prompt, model="gpt-4.1", temperature=0.7):
    """
    Calls the LLM to generate the configuration JSON.
    Returns: (config_dict, error_message)
    """
    try:
        print(f"[*] Calling {model} to generate configuration...")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content
        
        # Parse JSON
        config_dict = json.loads(raw_content)
        
        # Validate required keys
        required_keys = [
            "scenario_instructions", "generator_model", "judge_model",
            "dynamic_components", "necessary_labels", "sufficient_labels",
            "dialogue_controls"
        ]
        
        missing_keys = [key for key in required_keys if key not in config_dict]
        if missing_keys:
            return None, f"Generated config is missing required keys: {missing_keys}"
        
        print("[✓] Configuration generated successfully!")
        return config_dict, None
        
    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON response: {e}"
    except Exception as e:
        return None, f"Error calling LLM: {e}"

def save_config(config_dict, scenario_name=None):
    """
    Saves the generated configuration to a JSON file.
    Returns: filepath
    """
    if not os.path.exists(SCENARIOS_OUTPUT_DIR):
        os.makedirs(SCENARIOS_OUTPUT_DIR)
    
    # Generate filename
    if scenario_name:
        clean_name = scenario_name.lower().replace(" ", "_")
        filename = f"{clean_name}.json"
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"scenario_{timestamp}.json"
    
    filepath = os.path.join(SCENARIOS_OUTPUT_DIR, filename)
    
    # Save with pretty formatting
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"[✓] Configuration saved to: {filepath}")
    return filepath

def preview_config(config_dict):
    """Displays a summary of the generated configuration."""
    print("\n" + "="*60)
    print("CONFIGURATION PREVIEW")
    print("="*60)
    
    print(f"\n📋 Scenario: {config_dict.get('scenario_instructions', 'N/A')[:150]}...")
    
    print(f"\n👥 Dynamic Components:")
    for component, values in config_dict.get('dynamic_components', {}).items():
        print(f"  • {component}: {len(values)} options")
        print(f"    Examples: {', '.join(values[:2])}...")
    
    print(f"\n✓ Necessary Labels: {len(config_dict.get('necessary_labels', {}))}")
    for label, definition in config_dict.get('necessary_labels', {}).items():
        print(f"  • {label}: {definition[:80]}...")
    
    print(f"\n+ Sufficient Labels: {len(config_dict.get('sufficient_labels', {}))}")
    for label, definition in list(config_dict.get('sufficient_labels', {}).items())[:2]:
        print(f"  • {label}: {definition[:80]}...")
    
    dialogue_controls = config_dict.get('dialogue_controls', {})
    print(f"\n💬 Dialogue Controls:")
    print(f"  • Turn range: {dialogue_controls.get('min_turns', '?')}-{dialogue_controls.get('max_turns', '?')}")
    print(f"  • Roles: {', '.join(dialogue_controls.get('roles', []))}")
    
    seed_count = len(config_dict.get('seed_dialogue', []))
    print(f"\n🌱 Seed Dialogues: {seed_count} examples provided")
    
    print("\n" + "="*60 + "\n")

# --- Main Workflow ---

def main():
    print("="*60)
    print("SYNTHETIC DATA CONFIGURATION GENERATOR")
    print("="*60)
    print("\nThis tool will help you create a configuration file for")
    print("generating synthetic training conversations.\n")
    
    # Step 1: Get user's description
    print("📝 Describe what kind of conversations you want to generate.")
    print("   Be specific about:")
    print("   - The behavior or pattern you want to capture")
    print("   - Who the participants are (user role, assistant behavior)")
    print("   - The absence of governing constraints combined with common forms of model misalignment.")
    print("\nExample: 'I want to generate conversations where an AI assistant")
    print("         gives critical medical advice without any disclaimer, so I can")
    print("         train a classifier to detect this behavior.'\n")
    
    print("-" * 60)
    user_need = input("Your description: ").strip()
    
    if not user_need:
        print("[ERROR] You must provide a description. Exiting.")
        return
    
    # Optional: Get scenario name
    scenario_name = input("\nOptional - Give this scenario a name (or press Enter to auto-generate): ").strip()
    
    # Step 2: Load template and generate config
    print("\n" + "-"*60)
    template = load_prompt_template(CONFIG_GENERATOR_PROMPT)
    prompt = template.format(user_need_description=user_need)
    
    config_dict, error = call_llm_for_config(prompt, CONFIG_GENERATOR_MODEL, temperature=0.7)
    
    if error:
        print(f"[ERROR] Failed to generate configuration: {error}")
        return
    
    # Step 3: Preview the configuration
    preview_config(config_dict)
    
    # Step 4: Ask for confirmation
    confirm = input("Does this configuration look good? (yes/no/edit): ").lower().strip()
    
    if confirm in ['n', 'no']:
        print("\n[*] Configuration discarded. Please run the script again with a clearer description.")
        return
    
    if confirm in ['e', 'edit']:
        print("\n[*] Opening manual edit mode...")
        print("    (In a real implementation, you could allow field-by-field editing)")
        print("    For now, the config will be saved and you can manually edit the JSON file.")
    
    # Step 5: Save the configuration
    filepath = save_config(config_dict, scenario_name)
    
    # Step 6: Offer to run the generation pipeline
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"\n1. Your configuration has been saved to: {filepath}")
    print(f"\n2. To generate conversations using this config, run:")
    print(f"   python run_gen.py --config {filepath}")
    print(f"\n3. Or manually edit the JSON file first if needed.")
    
    run_now = input("\nWould you like to start generating conversations now? (yes/no): ").lower().strip()
    
    if run_now in ['y', 'yes']:
        print("\n[*] Launching generation pipeline...")
        print(f"[*] This would execute: python run_gen.py --config {filepath}")
        print("[!] (Integration with run_gen.py needed - see instructions below)")
        # In practice, you would:
        # import run_gen
        # run_gen.main(config_path=filepath)
    else:
        print("\n[✓] Configuration ready! Run the pipeline when you're ready.")

if __name__ == "__main__":
    main()
