# Changelog

All notable changes to PDFGraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-26

### Added
- Automatic text extraction from PDFs
- Multilingual language detection (German, English, French, Spanish, Italian)
- Semantic similarity calculation using TF-IDF and Cosine similarity
- Automatic document clustering
- Interactive HTML network visualization with pyvis
- Double-click functionality to open PDF files from the visualization
- Detailed cluster analysis output
- Configurable parameters:
  - `similarity_threshold` (0.0-1.0) - Connection strength
  - `n_clusters` - Number of clusters
  - `multilingual` - Enable/disable multilingual processing
  - `repulsion_strength` (1.0-5.0) - Node repulsion force
  - `spring_k` (3.0-15.0) - Overall node distance
- Color-coded edges based on similarity strength:
  - Green edges for high similarity (>0.7)
  - Blue edges for medium similarity (>0.5)
  - Gray edges for low similarity (>0.3)
- Interactive controls: drag & drop, zoom, hover tooltips, multi-select
- Language distribution statistics in console output
- Comprehensive README documentation with usage examples

### Technical Details
- Python 3.7+ compatible
- Dependencies: PyPDF2, networkx, matplotlib, scikit-learn, numpy, pyvis, langdetect
- Performance: ~5-10 seconds for 10-20 PDFs, ~30 seconds for 50+ PDFs

[1.0.0]: https://github.com/Baum64/PDFGraph/releases/tag/v1.0.0
