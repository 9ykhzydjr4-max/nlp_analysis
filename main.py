"""Analyse-Porzess: Vorverarbeitung -> Vektorisierung -> Themenextraktion.

Aufruf:
    python main.py            # alle drei Schritte
    python main.py --ab 2     # erst ab Schritt 2 starten (Vorverarbeitung überspringen)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent

# Skripte für Verarbeitung
SKRIPTE = [
    "01_vorverarbeitung.py",
    "02_vektorisierung.py",
    "03_themenextraktion.py",
]

def skript_ausfuehren(nummer, name):
    # Skript-Ausführung aus, bei Fehler abbruch
    pfad = PROJECT / name
    if not pfad.exists():
        print(f"FEHLER: {name} wurde nicht gefunden.", flush=True)
        sys.exit(1)

    print(f"\n{'=' * 60}\nSchritt {nummer}: {name}\n{'=' * 60}", flush=True)
    start = time.time()
    ergebnis = subprocess.run([sys.executable, str(pfad)], cwd=PROJECT)
    dauer = (time.time() - start) / 60

    if ergebnis.returncode != 0:
        print(f"\nFEHLER in {name} (nach {dauer:.1f} min). Abbruch.", flush=True)
        sys.exit(ergebnis.returncode)
    print(f"Schritt {nummer} fertig nach {dauer:.1f} min.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="NLP-Analyse der Beschwerdetexte")
    parser.add_argument("--ab", type=int, default=1, choices=[1, 2, 3],
                        help="Ab welchem Schritt gestartet wird (Standard: 1)")
    argumente = parser.parse_args()

    gesamt_start = time.time()
    for nummer, name in enumerate(SKRIPTE, start=1):
        if nummer >= argumente.ab:
            skript_ausfuehren(nummer, name)

    print(f"\nAlle Schritte erfolgreich. Gesamtdauer: "
          f"{(time.time() - gesamt_start) / 60:.1f} min", flush=True)
    print(f"Ergebnisse im Ordner: {PROJECT / 'output'}", flush=True)


if __name__ == "__main__":
    main()