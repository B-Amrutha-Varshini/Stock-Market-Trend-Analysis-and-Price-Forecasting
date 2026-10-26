"""
Master runner script for the Stock Market Intelligence Platform.
Run this from the project root directory:
    python run_all.py
"""

import subprocess
import sys
import os

# Fix Unicode output on Windows consoles (cp1252 -> utf-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

scripts = [
    ("Data Fetch",          "src/data_fetch.py"),
    ("Data Validator",      "src/data_validator.py"),
    ("Data Processor",      "src/data_process.py"),
    ("Feature Engineering", "src/feature_engineering.py"),
    ("EDA Analysis",        "src/eda_analysis.py"),
    ("Risk Analysis",       "src/risk_analysis.py"),
    ("ARIMA Forecast",      "src/arima_forecast.py"),
    ("Prophet Forecast",    "src/prophet_forecast.py"),
    ("LSTM Forecast",       "src/lstm_forecast.py"),
]

print("\n" + "=" * 70)
print("  STOCK MARKET INTELLIGENCE PLATFORM - MASTER RUNNER")
print("=" * 70)
print(f"Project root : {PROJECT_ROOT}")
print(f"Python       : {PYTHON}")
print()

results = []

for name, script in scripts:
    script_path = os.path.join(PROJECT_ROOT, script)
    print(f"\n{'='*70}")
    print(f"  RUNNING: {name}")
    print(f"  Script : {script}")
    print(f"{'='*70}")

    result = subprocess.run(
        [PYTHON, script_path],
        cwd=PROJECT_ROOT,
        capture_output=False,
        text=True
    )

    if result.returncode == 0:
        status = "SUCCESS"
        print(f"\n[OK] {name} completed successfully.")
    else:
        status = "FAILED"
        print(f"\n[ERROR] {name} failed with exit code {result.returncode}.")

    results.append((name, script, status))

print("\n" + "=" * 70)
print("  FINAL EXECUTION SUMMARY")
print("=" * 70)

for name, script, status in results:
    indicator = "[OK]  " if status == "SUCCESS" else "[FAIL]" 
    print(f"  {indicator} {name:<25} {status}")

print("=" * 70)

failed = [r for r in results if r[2] == "FAILED"]
if failed:
    print(f"\n{len(failed)} script(s) failed. Check output above for details.")
    sys.exit(1)
else:
    print("\nAll scripts completed successfully!")
    print("\nTo start the dashboard:")
    print(f"  streamlit run {os.path.join(PROJECT_ROOT, 'dashboard', 'app.py')}")
