# PDFGraph - Mehrsprachige PDF-Netzwerk-Analyse

Ein Python-Tool zur semantischen Analyse und interaktiven Visualisierung von PDF-Dokumentsammlungen.

## Features

- ✅ Automatische Textextraktion aus PDFs
- ✅ Mehrsprachige Spracherkennung (Deutsch, Englisch, Französisch, Spanisch, Italienisch)
- ✅ Semantische Ähnlichkeitsberechnung (TF-IDF + Cosinus-Ähnlichkeit)
- ✅ Automatisches Dokumenten-Clustering
- ✅ Interaktive HTML-Netzwerk-Visualisierung
- ✅ Doppelklick zum Öffnen von PDFs
- ✅ Detaillierte Cluster-Analyse

## Installation

### Voraussetzungen
- Python 3.7+
- pip

### Dependencies installieren

```bash
pip install -r requirements.txt
```

**requirements.txt Inhalt:**
```
PyPDF2
networkx
matplotlib
scikit-learn
numpy
pyvis
langdetect
```

## Verwendung

### Basis-Beispiel

```python
python PDFGraph.py
```

Die Standardeinstellung verwendet den Ordner:
```
C:\Users\sebas\Nextcloud\Familienordner\Bücher\Naturwissenschaften
```

### Mit benutzerdefinierten Parametern

```python
from PDFGraph import main

main(
    folder_path="./meine_pdfs",
    similarity_threshold=0.25,      # Verbindungsstärke (0.0-1.0)
    n_clusters=3,                    # Anzahl der Cluster
    multilingual=True,               # Mehrsprachig aktivieren
    repulsion_strength=2.5,          # Knoten-Abstoßung (1.0-5.0)
    spring_k=10.0                    # Gesamtabstand (3.0-15.0)
)
```

## Output

Das Programm erzeugt:
- **network_interactive.html** - Interaktive Netzwerk-Visualisierung
- **Konsolen-Output** - Sprachverteilung, Cluster-Analyse, Ähnlichkeitswerte

## Bedienung der HTML-Visualisierung

| Aktion | Beschreibung |
|--------|-------------|
| **Doppelklick** | Öffnet die PDF-Datei |
| **Drag & Drop** | Knoten verschieben |
| **Zoom** | Mausrad zum Zoomen |
| **Hover** | Zeigt Ähnlichkeitswert an |
| **Multiselect** | Mehrere Knoten auswählen |

## Parameter-Tuning

### repulsion_strength (1.0 - 5.0)
- Steuert die Abstoßung schwacher Verbindungen
- Niedrig (1.0): Kompakte Darstellung
- Hoch (5.0): Starke Streuung

### spring_k (3.0 - 15.0)
- Abstand zwischen allen Knoten
- Niedrig (3.0): Kompaktes Netzwerk
- Hoch (15.0): Weitläufige Verteilung

### similarity_threshold (0.0 - 1.0)
- Minimale Ähnlichkeit für Verbindungen
- Niedrig (0.1): Viele Verbindungen
- Hoch (0.5): Nur starke Verbindungen

## Farbcodierung

- **Cluster** → Verschiedene Knotenfarben
- **Kanten grün** → Hohe Ähnlichkeit (>0.7)
- **Kanten blau** → Mittlere Ähnlichkeit (>0.5)
- **Kanten grau** → Schwache Ähnlichkeit (>0.3)

## Unterstützte Sprachen

- 🇩🇪 Deutsch
- 🇬🇧 Englisch
- 🇫🇷 Französisch
- 🇪🇸 Spanisch
- 🇮🇹 Italienisch

Mehrsprachige Dokumentsammlungen werden automatisch erkannt und verarbeitet.

## Beispiel-Workflow

```bash
# 1. PDFs in einen Ordner kopieren
mkdir ./meine_pdfs
cp *.pdf ./meine_pdfs/

# 2. PDFGraph ausführen
python PDFGraph.py

# 3. Generierte HTML-Datei öffnen
network_interactive.html
```

## Fehlerbehebung

**"Keine PDFs gefunden"**
→ Stelle sicher, dass der PDF-Ordner existiert und .pdf-Dateien enthält

**"pyvis nicht installiert"**
→ Führe aus: `pip install pyvis`

**"Kein Text extrahiert"**
→ Die PDF könnte gescannt/bilderbasiert sein (OCR erforderlich)

## Performance

- ~5-10 Sekunden für 10-20 PDFs
- ~30 Sekunden für 50+ PDFs
- Abhängig von Textmenge und CPU

## Lizenz

Frei verwendbar für private und kommerzielle Projekte.

## Kontakt / Support

Bei Fragen oder Problemen: Code prüfen und Fehlerausgaben analysieren.

---

**Version:** 1.0 | **Autor:** Sebastian Meyer | **Datum:** Oktober 2025
