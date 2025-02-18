import typing
from typing import List, Optional, Dict
import re
import sys
from datetime import datetime
from collections import defaultdict

def analyze_logs(log_file: str) -> Dict[str, List[str]]:
    error_patterns: Dict[str, List[str]] = defaultdict(list)
    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            logs: List[str] = file.readlines()
            
        for line in logs:
            # Look for common error indicators
            if any(indicator in line.lower() for indicator in ['error', 'fail', 'crash', 'exception', 'critical']):
                timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}', line)
                timestamp = timestamp_match.group(0) if timestamp_match else "Unknown Time"
                
                # Categorize errors
                if 'memory' in line.lower():
                    error_patterns['Memory Issues'].append(f"{timestamp}: {line.strip()}")
                elif 'cpu' in line.lower():
                    error_patterns['CPU Issues'].append(f"{timestamp}: {line.strip()}")
                elif 'disk' in line.lower():
                    error_patterns['Disk Issues'].append(f"{timestamp}: {line.strip()}")
                elif 'network' in line.lower():
                    error_patterns['Network Issues'].append(f"{timestamp}: {line.strip()}")
                else:
                    error_patterns['Other Errors'].append(f"{timestamp}: {line.strip()}")
                    
    except FileNotFoundError:
        print(f"Error: Could not find log file: {log_file}")
        return {}
    except Exception as e:
        print(f"Error analyzing logs: {str(e)}")
        return {}
    
    return error_patterns

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python script.py <log_file>")
        sys.exit(1)
        
    log_file: str = sys.argv[1]
    error_patterns: Dict[str, List[str]] = analyze_logs(log_file)
    
    if not error_patterns:
        print("No errors found or analysis failed.")
        sys.exit(1)
        
    print("\nCrash Analysis Report:")
    print("=" * 50)
    
    for category, errors in error_patterns.items():
        if errors:
            print(f"\n{category}:")
            print("-" * 30)
            for error in errors[-5:]:  # Show last 5 errors in each category
                print(error)
                
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
