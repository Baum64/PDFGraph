# PDFGraph - Multilingual PDF Network Analysis

A Python tool for semantic analysis and interactive visualization of PDF document collections.

## Features

- ✅ Automatic text extraction from PDFs
- ✅ Multilingual language detection (German, English, French, Spanish, Italian)
- ✅ Semantic similarity calculation (TF-IDF + Cosine similarity)
- ✅ Automatic document clustering
- ✅ Interactive HTML network visualization
- ✅ Double-click to open PDFs
- ✅ Detailed cluster analysis

## Installation

### Requirements
- Python 3.7+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt Content:**
```
PyPDF2
networkx
matplotlib
scikit-learn
numpy
pyvis
langdetect
```

## Usage With Custom Parameters

```python
from PDFGraph import main

main(
    folder_path="./my_pdfs",
    similarity_threshold=0.25,      # Connection strength (0.0-1.0)
    n_clusters=3,                    # Number of clusters
    multilingual=True,               # Enable multilingual processing
    repulsion_strength=2.5,          # Node repulsion (1.0-5.0)
    spring_k=10.0                    # Overall distance (3.0-15.0)
)
```

## Output

The program generates:
- **network_interactive.html** - Interactive network visualization
- **Console output** - Language distribution, cluster analysis, similarity values

## HTML Visualization Controls

| Action | Description |
|--------|-------------|
| **Double-click** | Opens the PDF file |
| **Drag & Drop** | Move nodes around |
| **Scroll** | Zoom in/out |
| **Hover** | Shows similarity value |
| **Multi-select** | Select multiple nodes |

## Parameter Tuning

### repulsion_strength (1.0 - 5.0)
- Controls repulsion of weak connections
- Low (1.0): Compact layout
- High (5.0): Strong dispersion

### spring_k (3.0 - 15.0)
- Distance between all nodes
- Low (3.0): Compact network
- High (15.0): Widespread distribution

### similarity_threshold (0.0 - 1.0)
- Minimum similarity for connections
- Low (0.1): Many connections
- High (0.5): Only strong connections

## Color Coding

- **Clusters** → Different node colors
- **Green edges** → High similarity (>0.7)
- **Blue edges** → Medium similarity (>0.5)
- **Gray edges** → Low similarity (>0.3)

## Supported Languages

- 🇩🇪 German
- 🇬🇧 English
- 🇫🇷 French
- 🇪🇸 Spanish
- 🇮🇹 Italian

Multilingual document collections are automatically detected and processed.

## Example Workflow

```bash
# 1. Copy PDFs to a folder
mkdir ./my_pdfs
cp *.pdf ./my_pdfs/

# 2. Run PDFGraph
python PDFGraph.py

# 3. Open the generated HTML file
network_interactive.html
```

## Troubleshooting

**"No PDFs found"**
→ Make sure the PDF folder exists and contains .pdf files

**"pyvis not installed"**
→ Run: `pip install pyvis`

**"No text extracted"**
→ The PDF might be scanned/image-based (OCR required)

## Performance

- ~5-10 seconds for 10-20 PDFs
- ~30 seconds for 50+ PDFs
- Depends on text volume and CPU

---

**Version:** 1.0 | **Author:** Sebastian Meyer | **Date:** October 2025
