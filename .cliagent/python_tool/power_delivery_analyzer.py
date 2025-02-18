import sys
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

class LogAnalyzer:
    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.power_keywords: List[str] = [
            'voltage', 'power', 'under-voltage', 
            'current', 'supply', 'shutdown', 
            'thermal', 'temperature', 'watts'
        ]
        self.critical_keywords: List[str] = [
            'error', 'critical', 'failure', 
            'warning', 'emergency', 'alert'
        ]
        self.findings: Dict[str, List[str]] = {
            'power_issues': [],
            'critical_events': [],
            'thermal_events': []
        }

    def read_logs(self) -> Optional[List[str]]:
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return f.readlines()
        except Exception as e:
            print(f"Error reading log file: {e}")
            return None

    def analyze_line(self, line: str) -> None:
        line_lower = line.lower()
        
        # Check for power-related issues
        if any(keyword in line_lower for keyword in self.power_keywords):
            self.findings['power_issues'].append(line.strip())
        
        # Check for critical events
        if any(keyword in line_lower for keyword in self.critical_keywords):
            self.findings['critical_events'].append(line.strip())
        
        # Check specifically for thermal issues
        if 'temp' in line_lower or 'thermal' in line_lower:
            self.findings['thermal_events'].append(line.strip())

    def analyze_logs(self) -> Dict[str, List[str]]:
        lines = self.read_logs()
        if not lines:
            return self.findings
        
        for line in lines:
            self.analyze_line(line)
        
        return self.findings

    def print_report(self) -> None:
        print("\n=== RPI5 Crash Analysis Report ===")
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log File: {self.log_file}\n")

        for category, events in self.findings.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            if events:
                for idx, event in enumerate(events, 1):
                    print(f"{idx}. {event}")
            else:
                print("No issues found.")

def main() -> None:
    log_file = Path('recent_logs.txt')
    if not log_file.exists():
        print(f"Error: {log_file} not found!")
        sys.exit(1)

    analyzer = LogAnalyzer(log_file)
    analyzer.analyze_logs()
    analyzer.print_report()

if __name__ == "__main__":
    main()
