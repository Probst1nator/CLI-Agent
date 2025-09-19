#!/usr/bin/env python3
"""
TTS Configuration Utility

Simple command-line utility to configure TTS settings using the global configuration system.

Usage:
    python configure_tts.py                    # Interactive configuration
    python configure_tts.py --show             # Show current settings
    python configure_tts.py --set-voice nova   # Set default voice
    python configure_tts.py --set-speed 1.2    # Set default speed
    python configure_tts.py --disable piper    # Disable an engine
    python configure_tts.py --enable piper     # Enable an engine
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.globals import g
from termcolor import colored

def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{colored('═' * 60, 'cyan')}")
    print(f"{colored(title.center(60), 'cyan', attrs=['bold'])}")
    print(f"{colored('═' * 60, 'cyan')}")

def print_subsection(title: str):
    """Print a formatted subsection."""
    print(f"\n{colored('▶ ' + title, 'white', attrs=['bold'])}")
    print(f"{colored('─' * (len(title) + 2), 'white')}")

def show_current_settings():
    """Display current TTS configuration."""
    print_section_header("Current TTS Configuration")
    
    print_subsection("Default Settings")
    settings_table = [
        ("Voice", g.TTS_DEFAULT_VOICE),
        ("Speed", f"{g.TTS_DEFAULT_SPEED}x"),
        ("Volume", f"{g.TTS_DEFAULT_VOLUME:.1f} ({int(g.TTS_DEFAULT_VOLUME * 100)}%)"),
        ("Language", g.TTS_DEFAULT_LANGUAGE),
        ("Gender", g.TTS_DEFAULT_GENDER),
        ("Quality", g.TTS_DEFAULT_QUALITY)
    ]
    
    for setting, value in settings_table:
        print(f"  {colored(setting + ':', 'yellow'):<12} {colored(str(value), 'white')}")
    
    print_subsection("Engine Priority")
    active_engines = g.get_tts_engine_priority()
    for i, engine in enumerate(active_engines, 1):
        status_color = "green"
        print(f"  {colored(f'{i}.', 'cyan')} {colored(engine, status_color)}")
    
    if g.TTS_DISABLED_ENGINES:
        print_subsection("Disabled Engines")
        for engine in g.TTS_DISABLED_ENGINES:
            print(f"  {colored('✗', 'red')} {colored(engine, 'red')}")
    
    print_subsection("Context Settings")
    for context, settings in g.TTS_CONTEXT_SETTINGS.items():
        voice = settings.get('voice', 'default')
        speed = settings.get('speed', 1.0)
        volume = settings.get('volume', 0.8)
        print(f"  {colored(context + ':', 'yellow'):<20} voice={colored(voice, 'white')}, speed={colored(f'{speed}x', 'white')}, volume={colored(f'{volume:.1f}', 'white')}")
    
    print(f"\n{colored('─' * 60, 'cyan')}")

def interactive_configuration():
    """Interactive TTS configuration."""
    print_section_header("TTS Configuration Wizard")
    
    while True:
        print_subsection("Configuration Menu")
        
        menu_options = [
            ("1", "🔍 Show current settings", "View all TTS configuration"),
            ("2", "🎙️  Set default voice", "Change the default voice"),
            ("3", "⚡ Set default speed", "Adjust speech speed"),
            ("4", "🔊 Set default volume", "Adjust audio volume"),
            ("5", "🔧 Configure engine priority", "Reorder TTS engines"),
            ("6", "⚙️  Enable/disable engines", "Turn engines on/off"),
            ("7", "📝 Configure context settings", "Per-context voice settings"),
            ("8", "🧪 Test current settings", "Test your configuration"),
            ("q", "🚪 Quit", "Exit configuration")
        ]
        
        # Print menu with better formatting
        for option, title, description in menu_options:
            print(f"  {colored(option + '.', 'cyan')} {colored(title, 'white', attrs=['bold'])}")
            print(f"      {colored(description, 'white')}")
        
        print(f"\n{colored('─' * 50, 'cyan')}")
        choice = input(f"{colored('Choose option [1-8, q]:', 'yellow')} ").strip().lower()
        
        try:
            if choice == 'q':
                print(f"\n{colored('👋 Configuration saved! Goodbye!', 'green', attrs=['bold'])}")
                break
            elif choice == '1':
                show_current_settings()
            elif choice == '2':
                configure_default_voice()
            elif choice == '3':
                configure_default_speed()
            elif choice == '4':
                configure_default_volume()
            elif choice == '5':
                configure_engine_priority()
            elif choice == '6':
                configure_engine_enable_disable()
            elif choice == '7':
                configure_context_settings()
            elif choice == '8':
                test_current_settings()
            else:
                print(f"\n{colored('❌ Invalid option. Please choose 1-8 or q.', 'red')}")
                
            if choice != 'q':
                input(f"\n{colored('Press Enter to continue...', 'cyan')}")
                
        except KeyboardInterrupt:
            print(f"\n\n{colored('👋 Configuration interrupted. Settings saved!', 'yellow')}")
            break
        except Exception as e:
            print(f"\n{colored(f'❌ Error: {e}', 'red')}")
            input(f"{colored('Press Enter to continue...', 'cyan')}")

def configure_default_voice():
    """Configure default voice."""
    print_subsection("🎙️ Configure Default Voice")
    print(f"  {colored('Current:', 'yellow')} {colored(g.TTS_DEFAULT_VOICE, 'white', attrs=['bold'])}")
    
    print(f"\n{colored('Available Voice Options:', 'white', attrs=['bold'])}")
    voice_categories = [
        ("General", ["default", "female", "male"]),
        ("OpenAI", ["nova", "alloy", "echo", "fable", "onyx", "shimmer"]),
        ("ElevenLabs", ["rachel", "drew", "clyde", "dave"]),
        ("Piper", ["german_male", "german_female", "english_male", "english_female"])
    ]
    
    for category, voices in voice_categories:
        print(f"  {colored(category + ':', 'cyan')} {', '.join(colored(v, 'white') for v in voices)}")
    
    print(f"\n{colored('─' * 40, 'cyan')}")
    new_voice = input(f"{colored('Enter new default voice:', 'yellow')} ").strip()
    
    if new_voice:
        g.save_tts_defaults(voice=new_voice)
        print(f"\n{colored('✅ Success!', 'green', attrs=['bold'])} Default voice set to: {colored(new_voice, 'white', attrs=['bold'])}")
    else:
        print(f"\n{colored('⚠️ No changes made.', 'yellow')}")

def configure_default_speed():
    """Configure default speech speed."""
    print_subsection("⚡ Configure Default Speed")
    print(f"  {colored('Current:', 'yellow')} {colored(f'{g.TTS_DEFAULT_SPEED}x', 'white', attrs=['bold'])}")
    
    print(f"\n{colored('Speed Guide:', 'white', attrs=['bold'])}")
    speed_guide = [
        ("0.5x", "Very slow (good for learning)"),
        ("0.8x", "Slow (clear pronunciation)"),
        ("1.0x", "Normal (default speed)"),
        ("1.2x", "Slightly fast (efficient)"),
        ("1.5x", "Fast (quick reading)"),
        ("2.0x", "Very fast (speed reading)")
    ]
    
    for speed, desc in speed_guide:
        print(f"  {colored(speed, 'cyan'):<6} - {colored(desc, 'white')}")
    
    print(f"\n{colored('Valid range: 0.25 to 4.0', 'white')}")
    print(f"{colored('─' * 40, 'cyan')}")
    
    try:
        new_speed_str = input(f"{colored('Enter new speed [0.25-4.0]:', 'yellow')} ").strip()
        if new_speed_str:
            new_speed = float(new_speed_str)
            if 0.25 <= new_speed <= 4.0:
                g.save_tts_defaults(speed=new_speed)
                print(f"\n{colored('✅ Success!', 'green', attrs=['bold'])} Default speed set to: {colored(f'{new_speed}x', 'white', attrs=['bold'])}")
            else:
                print(f"\n{colored('❌ Error:', 'red')} Speed must be between 0.25 and 4.0")
        else:
            print(f"\n{colored('⚠️ No changes made.', 'yellow')}")
    except ValueError:
        print(f"\n{colored('❌ Error:', 'red')} Invalid speed value. Please enter a number.")

def configure_default_volume():
    """Configure default volume level."""
    print_subsection("🔊 Configure Default Volume")
    print(f"  {colored('Current:', 'yellow')} {colored(f'{g.TTS_DEFAULT_VOLUME:.1f}', 'white', attrs=['bold'])} ({colored(f'{int(g.TTS_DEFAULT_VOLUME * 100)}%', 'white', attrs=['bold'])})")
    
    print(f"\n{colored('Volume Guide:', 'white', attrs=['bold'])}")
    volume_guide = [
        ("0.1", "10%", "Very quiet"),
        ("0.3", "30%", "Quiet"),
        ("0.5", "50%", "Medium-low"),
        ("0.8", "80%", "Good (recommended)"),
        ("1.0", "100%", "Maximum volume")
    ]
    
    for vol, percent, desc in volume_guide:
        print(f"  {colored(vol, 'cyan'):<5} ({colored(percent, 'white'):<4}) - {colored(desc, 'white')}")
    
    print(f"\n{colored('Valid range: 0.1 to 1.0', 'white')}")
    print(f"{colored('─' * 40, 'cyan')}")
    
    try:
        new_volume_str = input(f"{colored('Enter new volume [0.1-1.0]:', 'yellow')} ").strip()
        if new_volume_str:
            new_volume = float(new_volume_str)
            if 0.1 <= new_volume <= 1.0:
                g.save_tts_defaults(volume=new_volume)
                percent = int(new_volume * 100)
                print(f"\n{colored('✅ Success!', 'green', attrs=['bold'])} Default volume set to: {colored(f'{new_volume:.1f}', 'white', attrs=['bold'])} ({colored(f'{percent}%', 'white', attrs=['bold'])})")
            else:
                print(f"\n{colored('❌ Error:', 'red')} Volume must be between 0.1 and 1.0")
        else:
            print(f"\n{colored('⚠️ No changes made.', 'yellow')}")
    except ValueError:
        print(f"\n{colored('❌ Error:', 'red')} Invalid volume value. Please enter a number.")

def configure_engine_priority():
    """Configure TTS engine priority order."""
    print_subsection("🔧 Configure Engine Priority")
    
    current_engines = g.get_tts_engine_priority()
    print(f"  {colored('Current Priority Order:', 'yellow')}")
    for i, engine in enumerate(current_engines, 1):
        print(f"    {colored(f'{i}.', 'cyan')} {colored(engine, 'white')}")
    
    print(f"\n{colored('Available Engines:', 'white', attrs=['bold'])}")
    valid_engines = ["openai", "elevenlabs", "piper", "coqui", "huggingface", "system"]
    engine_descriptions = {
        "openai": "Premium quality API (requires key)",
        "elevenlabs": "Natural voices API (requires key)", 
        "piper": "High-quality offline TTS",
        "coqui": "Advanced multi-lingual TTS",
        "huggingface": "Modern transformer models",
        "system": "System TTS (espeak, say, etc.)"
    }
    
    for engine in valid_engines:
        desc = engine_descriptions.get(engine, "")
        print(f"  {colored(engine, 'cyan'):<12} - {colored(desc, 'white')}")
    
    print(f"\n{colored('Instructions:', 'white', attrs=['bold'])}")
    print("  • Enter engines in order of preference (comma-separated)")
    print("  • You don't need to include all engines")
    print("  • Example: piper,openai,system")
    
    print(f"\n{colored('─' * 50, 'cyan')}")
    new_order = input(f"{colored('Enter new priority order:', 'yellow')} ").strip()
    
    if new_order:
        # Clean up the input
        engines = [engine.strip().lower() for engine in new_order.split(',') if engine.strip()]
        
        # Check for typos and suggest corrections
        corrected_engines = []
        invalid_engines = []
        
        for engine in engines:
            if engine in valid_engines:
                corrected_engines.append(engine)
            else:
                # Try to find close matches for typos
                suggestion = find_closest_engine(engine, valid_engines)
                if suggestion:
                    print(f"\n{colored('❓ Did you mean:', 'yellow')} '{colored(engine, 'red')}' → '{colored(suggestion, 'green')}'?")
                    confirm = input(f"  Use '{suggestion}' instead? [Y/n]: ").strip().lower()
                    if confirm != 'n':
                        corrected_engines.append(suggestion)
                        print(f"  {colored('✓ Corrected', 'green')}: {engine} → {suggestion}")
                    else:
                        invalid_engines.append(engine)
                else:
                    invalid_engines.append(engine)
        
        # Remove duplicates while preserving order
        unique_engines = []
        for engine in corrected_engines:
            if engine not in unique_engines:
                unique_engines.append(engine)
        
        if invalid_engines:
            print(f"\n{colored('❌ Invalid engines:', 'red')} {', '.join(invalid_engines)}")
            print(f"  Valid options: {', '.join(colored(e, 'cyan') for e in valid_engines)}")
            
        if unique_engines:
            # Add any missing engines to the end to maintain completeness
            for engine in valid_engines:
                if engine not in unique_engines:
                    unique_engines.append(engine)
            
            g.set_tts_engine_priority(unique_engines)
            print(f"\n{colored('✅ Success!', 'green', attrs=['bold'])} Engine priority updated:")
            for i, engine in enumerate(unique_engines, 1):
                print(f"    {colored(f'{i}.', 'cyan')} {colored(engine, 'white')}")
        else:
            print(f"\n{colored('⚠️ No valid engines specified. Priority unchanged.', 'yellow')}")
    else:
        print(f"\n{colored('⚠️ No changes made.', 'yellow')}")

def find_closest_engine(input_engine: str, valid_engines: list) -> str:
    """Find the closest matching engine name for typo correction."""
    input_lower = input_engine.lower()
    
    # Direct substring matches
    for engine in valid_engines:
        if input_lower in engine or engine in input_lower:
            return engine
    
    # Levenshtein-like simple matching
    best_match = None
    min_distance = float('inf')
    
    for engine in valid_engines:
        distance = simple_edit_distance(input_lower, engine)
        if distance < min_distance and distance <= 2:  # Allow up to 2 character differences
            min_distance = distance
            best_match = engine
    
    return best_match

def simple_edit_distance(s1: str, s2: str) -> int:
    """Calculate simple edit distance between two strings."""
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    
    # Count character differences (simplified)
    diff_count = 0
    min_len = min(len(s1), len(s2))
    
    for i in range(min_len):
        if s1[i] != s2[i]:
            diff_count += 1
    
    diff_count += abs(len(s1) - len(s2))
    return diff_count

def configure_engine_enable_disable():
    """Enable or disable TTS engines."""
    print(f"\n{colored('Enable/Disable Engines', 'yellow')}")
    
    all_engines = ["openai", "elevenlabs", "piper", "coqui", "huggingface", "system"]
    disabled = g.TTS_DISABLED_ENGINES
    
    print("Current status:")
    for engine in all_engines:
        status = "disabled" if engine in disabled else "enabled"
        color = "red" if status == "disabled" else "green"
        print(f"  {engine}: {colored(status, color)}")
    
    action = input("\nEnable (e) or disable (d) an engine? [e/d]: ").strip().lower()
    if action in ['e', 'd']:
        engine = input("Enter engine name: ").strip()
        if engine in all_engines:
            if action == 'd':
                g.disable_tts_engine(engine)
                print(f"{colored('✓ Disabled engine:', 'yellow')} {engine}")
            else:
                g.enable_tts_engine(engine)
                print(f"{colored('✓ Enabled engine:', 'green')} {engine}")
        else:
            print(f"{colored('Invalid engine name', 'red')}")

def configure_context_settings():
    """Configure context-specific TTS settings."""
    print(f"\n{colored('Configure Context Settings', 'yellow')}")
    
    contexts = list(g.TTS_CONTEXT_SETTINGS.keys())
    print("Available contexts:")
    for i, context in enumerate(contexts, 1):
        settings = g.get_tts_context_settings(context)
        print(f"  {i}. {context}: voice={settings.get('voice', 'default')}, speed={settings.get('speed', 1.0)}")
    
    try:
        choice = int(input("Select context to configure (number): ")) - 1
        if 0 <= choice < len(contexts):
            context = contexts[choice]
            current = g.get_tts_context_settings(context)
            
            print(f"\nConfiguring {context}:")
            print(f"Current: voice={current.get('voice', 'default')}, speed={current.get('speed', 1.0)}")
            
            new_voice = input("New voice (or Enter to keep current): ").strip()
            new_speed_str = input("New speed (or Enter to keep current): ").strip()
            
            new_settings = current.copy()
            if new_voice:
                new_settings['voice'] = new_voice
            if new_speed_str:
                try:
                    new_settings['speed'] = float(new_speed_str)
                except ValueError:
                    print(f"{colored('Invalid speed, keeping current', 'yellow')}")
            
            g.set_tts_context_settings(context, new_settings)
            print(f"{colored('✓ Context settings updated', 'green')}")
        else:
            print(f"{colored('Invalid context selection', 'red')}")
    except ValueError:
        print(f"{colored('Invalid selection', 'red')}")

def test_current_settings():
    """Test current TTS settings."""
    print(f"\n{colored('Testing Current Settings', 'yellow')}")
    
    from shared.utils_audio import text_to_speech
    
    test_text = "This is a test of your current TTS configuration settings."
    print(f"Speaking: '{test_text}'")
    
    try:
        text_to_speech(
            text=test_text,
            voice=g.TTS_DEFAULT_VOICE,
            speed=g.TTS_DEFAULT_SPEED,
            play=True,
            context="test"
        )
        print(f"{colored('✓ Test completed', 'green')}")
    except Exception as e:
        print(f"{colored('✗ Test failed:', 'red')} {e}")

def main():
    parser = argparse.ArgumentParser(description="TTS Configuration Utility")
    parser.add_argument("--show", action="store_true", help="Show current settings")
    parser.add_argument("--set-voice", help="Set default voice")
    parser.add_argument("--set-speed", type=float, help="Set default speed")
    parser.add_argument("--set-volume", type=float, help="Set default volume (0.1-1.0)")
    parser.add_argument("--disable", help="Disable a TTS engine")
    parser.add_argument("--enable", help="Enable a TTS engine")
    parser.add_argument("--test", action="store_true", help="Test current settings")
    
    args = parser.parse_args()
    
    if args.show:
        show_current_settings()
    elif args.set_voice:
        g.save_tts_defaults(voice=args.set_voice)
        print(f"{colored('✅ Success!', 'green', attrs=['bold'])} Default voice set to: {colored(args.set_voice, 'white', attrs=['bold'])}")
    elif args.set_speed:
        if 0.25 <= args.set_speed <= 4.0:
            g.save_tts_defaults(speed=args.set_speed)
            print(f"{colored('✅ Success!', 'green', attrs=['bold'])} Default speed set to: {colored(f'{args.set_speed}x', 'white', attrs=['bold'])}")
        else:
            print(f"{colored('❌ Error:', 'red')} Speed must be between 0.25 and 4.0")
    elif args.set_volume:
        if 0.1 <= args.set_volume <= 1.0:
            g.save_tts_defaults(volume=args.set_volume)
            percent = int(args.set_volume * 100)
            print(f"{colored('✅ Success!', 'green', attrs=['bold'])} Default volume set to: {colored(f'{args.set_volume:.1f}', 'white', attrs=['bold'])} ({colored(f'{percent}%', 'white', attrs=['bold'])})")
        else:
            print(f"{colored('❌ Error:', 'red')} Volume must be between 0.1 and 1.0")
    elif args.disable:
        g.disable_tts_engine(args.disable)
        print(f"{colored('⚠️ Engine disabled:', 'yellow')} {colored(args.disable, 'white', attrs=['bold'])}")
    elif args.enable:
        g.enable_tts_engine(args.enable)
        print(f"{colored('✅ Engine enabled:', 'green')} {colored(args.enable, 'white', attrs=['bold'])}")
    elif args.test:
        test_current_settings()
    else:
        interactive_configuration()

if __name__ == "__main__":
    main()