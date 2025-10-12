import PyPDF2
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import os
from pathlib import Path
import re

# Mehrsprachige Stoppwörter (erweitert)
STOPWORDS = {
    'de': ['der', 'die', 'das', 'und', 'oder', 'aber', 'ist', 'sind', 'ein', 'eine', 
           'werden', 'wurde', 'mit', 'von', 'zu', 'im', 'am', 'für', 'auf', 'durch',
           'bei', 'als', 'nach', 'über', 'unter', 'zwischen', 'hat', 'haben', 'wird',
           'dass', 'wenn', 'weil', 'sich', 'noch', 'auch', 'nur', 'kann', 'können',
           'dem', 'den', 'des', 'einem', 'eines', 'einer', 'nicht', 'war', 'waren'],
    'en': ['the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
           'of', 'with', 'by', 'from', 'as', 'an', 'be', 'has', 'have', 'had',
           'that', 'which', 'this', 'these', 'those', 'will', 'would', 'can', 'could',
           'was', 'were', 'been', 'being', 'am', 'not', 'there', 'their', 'they'],
    'fr': ['le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'est', 'sont',
           'dans', 'pour', 'avec', 'sur', 'qui', 'que', 'au', 'aux', 'ce', 'cette',
           'ces', 'mais', 'pas', 'ne', 'se', 'si', 'nous', 'vous', 'ils', 'elles'],
    'es': ['el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'y', 'es', 'son',
           'en', 'por', 'para', 'con', 'que', 'este', 'esta', 'estos', 'estas',
           'no', 'se', 'al', 'lo', 'como', 'pero', 'más', 'su', 'sus', 'ha', 'han'],
    'it': ['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'una', 'di', 'da', 'e', 'è',
           'sono', 'in', 'per', 'con', 'su', 'che', 'questo', 'questa', 'questi',
           'non', 'si', 'al', 'alla', 'agli', 'alle', 'del', 'della', 'dei', 'delle']
}

def extract_text_from_pdf(pdf_path):
    """Extrahiert Text aus einer PDF-Datei"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
    except Exception as e:
        print(f"Fehler beim Lesen von {pdf_path}: {e}")
    return text.strip()

def detect_language(text):
    """Robuste Spracherkennung mit langdetect und Fallback"""
    try:
        # Versuche langdetect (am zuverlässigsten)
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # Für konsistente Ergebnisse
        
        # Nutze nur ersten Teil des Textes für schnellere Erkennung
        sample = text[:3000] if len(text) > 3000 else text
        detected = detect(sample)
        
        # Mappe auf unsere unterstützten Sprachen
        if detected in ['de', 'en', 'fr', 'es', 'it']:
            return detected
        return 'en'  # Fallback für andere Sprachen
        
    except ImportError:
        # Fallback: Verbesserte manuelle Erkennung mit Zeichenhäufigkeit
        return detect_language_manual(text)
    except:
        # Bei Fehler: manuelle Erkennung
        return detect_language_manual(text)

def detect_language_manual(text):
    """Verbesserte manuelle Spracherkennung als Fallback"""
    if not text or len(text) < 50:
        return 'en'
    
    text_lower = text.lower()
    
    # Häufigkeitsanalyse charakteristischer Wörter (pro 1000 Zeichen)
    text_length = len(text_lower)
    
    # Deutsche Marker (inkl. Umlaute und typische Konstruktionen)
    de_markers = {
        'words': ['der', 'die', 'das', 'und', 'ist', 'werden', 'von', 'den', 'des', 'dem',
                  'ein', 'eine', 'einem', 'eines', 'nicht', 'auch', 'sich', 'mit', 'auf',
                  'für', 'sind', 'wird', 'haben', 'wurde', 'können', 'durch'],
        'chars': ['ä', 'ö', 'ü', 'ß']
    }
    
    # Englische Marker
    en_markers = {
        'words': ['the', 'and', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
                  'had', 'will', 'would', 'can', 'could', 'this', 'that', 'with', 'for',
                  'from', 'they', 'their', 'which', 'when', 'where', 'what'],
        'chars': []
    }
    
    # Französische Marker
    fr_markers = {
        'words': ['le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'est', 'sont',
                  'dans', 'pour', 'avec', 'sur', 'qui', 'que', 'cette', 'ces', 'pas'],
        'chars': ['é', 'è', 'ê', 'à', 'ç', 'ù']
    }
    
    # Spanische Marker
    es_markers = {
        'words': ['el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'y', 'es', 'son',
                  'en', 'por', 'para', 'con', 'que', 'este', 'esta', 'estos', 'no'],
        'chars': ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']
    }
    
    scores = {}
    
    for lang, markers in [('de', de_markers), ('en', en_markers), 
                          ('fr', fr_markers), ('es', es_markers)]:
        score = 0
        
        # Wort-Score (häufige Wörter als ganze Wörter)
        for word in markers['words']:
            # Verwende Wortgrenzen für genauere Matches
            import re
            pattern = r'\b' + re.escape(word) + r'\b'
            count = len(re.findall(pattern, text_lower))
            score += count
        
        # Zeichen-Score (sprachspezifische Zeichen)
        for char in markers['chars']:
            score += text_lower.count(char) * 3  # Gewichte Sonderzeichen höher
        
        # Normalisiere auf Textlänge
        scores[lang] = (score / text_length) * 1000
    
    # Wähle Sprache mit höchstem Score
    if scores:
        detected_lang = max(scores, key=scores.get)
        
        # Confidence-Check: Mindestunterschied erforderlich
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 1.5:
            return detected_lang
        elif sorted_scores[0] > 5:  # Absoluter Mindestscore
            return detected_lang
    
    return 'en'  # Default Fallback

def get_combined_stopwords():
    """Kombiniert deutsche und englische Stoppwörter"""
    combined = set()
    for words in STOPWORDS.values():
        combined.update(words)
    return list(combined)

def load_pdfs_from_folder(folder_path):
    """Lädt alle PDFs aus einem Ordner und erkennt Sprache"""
    pdf_files = list(Path(folder_path).glob("*.pdf"))
    documents = {}
    languages = {}
    file_paths = {}  # Speichere vollständige Pfade
    
    print(f"Lade {len(pdf_files)} PDF-Dateien...")
    print("-" * 70)
    
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            lang = detect_language(text)
            documents[pdf_file.name] = text
            languages[pdf_file.name] = lang
            file_paths[pdf_file.name] = str(pdf_file.absolute())  # Absoluter Pfad
            
            # Zeige auch Confidence-Info
            lang_name = {'de': 'Deutsch', 'en': 'Englisch', 'fr': 'Französisch', 
                        'es': 'Spanisch', 'it': 'Italienisch'}.get(lang, lang)
            print(f"✓ {pdf_file.name[:50]:<50} | {len(text):>6} chars | {lang_name}")
        else:
            print(f"✗ {pdf_file.name} - Kein Text extrahiert")
    
    print("-" * 70)
    return documents, languages, file_paths

def calculate_similarity_matrix(documents, use_multilingual=True):
    """Berechnet Ähnlichkeitsmatrix mit mehrsprachiger Unterstützung"""
    doc_names = list(documents.keys())
    doc_texts = list(documents.values())
    
    # TF-IDF Vektorisierung mit mehrsprachigen Stoppwörtern
    if use_multilingual:
        stopwords = get_combined_stopwords()
    else:
        stopwords = 'english'
    
    vectorizer = TfidfVectorizer(
        max_features=1500,
        stop_words=stopwords,
        ngram_range=(1, 3),  # Erweitert auf Trigramme für bessere semantische Erfassung
        min_df=1,  # Auch seltene Begriffe berücksichtigen
        max_df=0.8,  # Zu häufige Begriffe ausschließen
        token_pattern=r'\b[a-zA-ZäöüÄÖÜß]{3,}\b'  # Deutsche Umlaute unterstützen
    )
    
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    
    # Kosinus-Ähnlichkeit berechnen
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return similarity_matrix, doc_names, vectorizer

def cluster_documents(similarity_matrix, n_clusters=3):
    """Clustert Dokumente basierend auf Ähnlichkeit"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(1 - similarity_matrix)
    return clusters

def create_network_graph(similarity_matrix, doc_names, clusters, threshold=0.3, repulsion_strength=2.0):
    """Erstellt Netzwerkgraph aus Ähnlichkeitsmatrix mit invertierten Gewichten
    
    Args:
        repulsion_strength: Verstärkungsfaktor für Abstoßung bei geringer Ähnlichkeit (1.0-5.0)
                           Höher = stärkere Abstoßung schwacher Verbindungen
    """
    G = nx.Graph()
    
    for i, name in enumerate(doc_names):
        G.add_node(i, label=name, cluster=int(clusters[i]))
    
    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            similarity = similarity_matrix[i][j]
            if similarity > threshold:
                # Invertiere Gewicht: hohe Ähnlichkeit = kurze Distanz
                # Spring-Layout interpretiert Gewicht als "gewünschte Länge"
                # weight = 1/similarity macht ähnliche Knoten näher
                # repulsion_strength verstärkt die Abstoßung bei geringer Ähnlichkeit
                edge_weight = (1.0 / (similarity + 0.1)) ** repulsion_strength
                G.add_edge(i, j, weight=edge_weight, similarity=similarity)
    
    return G

def create_interactive_html(G, doc_names, clusters, languages, file_paths, repulsion_strength=2.0, output_file="network_interactive.html"):
    """Erstellt interaktive HTML-Visualisierung mit korrekter Physik und klickbaren Knoten"""
    try:
        from pyvis.network import Network
        import urllib.parse
        
        # Berechne Cluster-Statistiken für bessere Farben
        n_clusters = len(set(clusters))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
        
        # Erstelle Netzwerk mit angepassten Parametern
        net = Network(
            height="900px", 
            width="100%", 
            bgcolor="#f8f9fa", 
            font_color="black",
            notebook=False
        )
        
        # Physik-Einstellungen
        base_spring_length = 200 * (1 + repulsion_strength / 2)
        
        net.barnes_hut(
            gravity=-15000 * repulsion_strength,
            central_gravity=0.01,
            spring_length=int(base_spring_length),
            damping=0.09,
            overlap=0
        )
        
        # Knoten hinzufügen mit onclick zum Öffnen der PDF
        for node in G.nodes():
            cluster_id = clusters[node]
            doc_name = doc_names[node]
            lang = languages.get(doc_name, '?').upper()
            color = colors[cluster_id % len(colors)]
            
            # Erstelle file:// URL
            file_path = file_paths.get(doc_name, '')
            file_url = 'file:///' + file_path.replace('\\', '/')
            
            # Tooltip ohne Link (da der ganze Knoten klickbar ist)
            tooltip_html = f"""{doc_name}Cluster: {cluster_id + 1}"""
            
            net.add_node(
                node,
                label=f"[{lang}] {doc_name[:40]}",
                title=tooltip_html,
                color=color,
                size=35,
                font={'size': 14, 'color': 'black'},
                borderWidth=3,
                borderWidthSelected=5,
                url=file_url  # Wird von pyvis als onclick verwendet
            )
        
        # Kanten hinzufügen
        for edge in G.edges(data=True):
            similarity = edge[2]['similarity']
            edge_weight = edge[2]['weight']
            
            length = int(edge_weight * 50)
            width = max(1, similarity * 8)
            
            if similarity > 0.7:
                color = {'color': '#2ecc71', 'opacity': 0.8}
            elif similarity > 0.5:
                color = {'color': '#3498db', 'opacity': 0.6}
            elif similarity > 0.3:
                color = {'color': '#95a5a6', 'opacity': 0.4}
            else:
                color = {'color': '#ecf0f1', 'opacity': 0.2}
            
            net.add_edge(
                edge[0], edge[1],
                value=width,
                length=length,
                title=f"Ähnlichkeit: {similarity:.2f}",
                color=color
            )
        
        # Konfigurations-Optionen
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 500,
              "updateInterval": 25
            },
            "barnesHut": {
              "avoidOverlap": 0.5
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 50,
            "hideEdgesOnDrag": true,
            "multiselect": true,
            "navigationButtons": true
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "continuous"
            }
          },
          "nodes": {
            "font": {
              "size": 14,
              "face": "Arial"
            }
          }
        }
        """)
        
        # Speichere Graph
        net.save_graph(output_file)
        
        # Modifiziere die HTML-Datei, um Doppelklick-Handler hinzuzufügen
        with open(output_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Füge JavaScript für Doppelklick-Handling hinzu
        custom_js = """
        <script type="text/javascript">
        // Event-Listener für Knoten-Doppelklick
        network.on("doubleClick", function(params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                if (node && node.url) {
                    // Öffne PDF in neuem Tab/Fenster
                    window.open(node.url, '_blank');
                }
            }
        });
        
        // Zeige Cursor-Hinweis beim Hover
        network.on("hoverNode", function(params) {
            document.body.style.cursor = 'pointer';
        });
        network.on("blurNode", function(params) {
            document.body.style.cursor = 'default';
        });
        </script>
        """
        
        # Füge Custom JS vor </body> ein
        html_content = html_content.replace('</body>', custom_js + '</body>')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Interaktive Visualisierung: {output_file}")
        print(f"  Physik-Parameter: gravity={-15000 * repulsion_strength}, spring_length={int(base_spring_length)}")
        print(f"  💡 Tipp: DOPPELKLICK auf einen Knoten öffnet die PDF-Datei")
        
    except ImportError:
        print("⚠ pyvis nicht installiert. Installation:")
        print("  pip install pyvis")

def analyze_clusters(doc_names, clusters, similarity_matrix, languages):
    """Analysiert Cluster mit Sprach-Informationen"""
    print("\n" + "="*70)
    print("CLUSTER-ANALYSE (Mehrsprachig)")
    print("="*70)
    
    n_clusters = len(set(clusters))
    
    for cluster_id in range(n_clusters):
        docs_in_cluster = [(doc_names[i], languages.get(doc_names[i], '?')) 
                          for i, c in enumerate(clusters) if c == cluster_id]
        
        print(f"\n📁 Cluster {cluster_id + 1} ({len(docs_in_cluster)} Dokumente):")
        
        # Sprach-Statistik
        lang_count = {}
        for doc, lang in docs_in_cluster:
            lang_count[lang] = lang_count.get(lang, 0) + 1
        
        print(f"   Sprachen: {', '.join([f'{lang.upper()}: {count}' for lang, count in lang_count.items()])}")
        print(f"   Dokumente:")
        
        for doc, lang in docs_in_cluster:
            print(f"     - [{lang.upper()}] {doc}")
        
        indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        if len(indices) > 1:
            cluster_similarities = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    cluster_similarities.append(similarity_matrix[indices[i]][indices[j]])
            avg_sim = np.mean(cluster_similarities)
            print(f"   Durchschn. Ähnlichkeit: {avg_sim:.3f}")

def main(folder_path, similarity_threshold=0.25, n_clusters=None, multilingual=True, 
         repulsion_strength=2.0, spring_k=8.0):
    """Hauptfunktion mit mehrsprachiger Unterstützung
    
    Args:
        folder_path: Pfad zum PDF-Ordner
        similarity_threshold: Minimale Ähnlichkeit für Verbindungen (0.0-1.0)
        n_clusters: Anzahl Cluster (None = automatisch)
        multilingual: Mehrsprachige Verarbeitung aktivieren
        repulsion_strength: Abstoßungsverstärkung für schwache Verbindungen (1.0-5.0)
        spring_k: Spring-Layout Abstandsparameter (3.0-15.0, höher = mehr Abstand)
    """
    
    print("MEHRSPRACHIGE PDF-CLUSTER-NETZWERK-ANALYSE")
    print("="*70)
    
    documents, languages, file_paths = load_pdfs_from_folder(folder_path)
    
    if len(documents) < 2:
        print("⚠ Mindestens 2 PDF-Dateien erforderlich!")
        return
    
    lang_summary = {}
    for lang in languages.values():
        lang_summary[lang] = lang_summary.get(lang, 0) + 1
    print(f"\n📊 Sprachverteilung: {', '.join([f'{k.upper()}: {v}' for k, v in lang_summary.items()])}")
    
    print("\nBerechne Ähnlichkeiten (mehrsprachig)...")
    similarity_matrix, doc_names, vectorizer = calculate_similarity_matrix(documents, multilingual)
    
    if n_clusters is None:
        n_clusters = min(max(2, len(documents) // 3), 6)
    
    print(f"Clustere Dokumente in {n_clusters} Gruppen...")
    clusters = cluster_documents(similarity_matrix, n_clusters)
    
    print("Erstelle Netzwerkgraph...")
    G = create_network_graph(similarity_matrix, doc_names, clusters, similarity_threshold, repulsion_strength)
    
    print(f"Graph: {G.number_of_nodes()} Knoten, {G.number_of_edges()} Kanten")
    print(f"Abstoßungs-Parameter: repulsion_strength={repulsion_strength}, spring_k={spring_k}")
    
    create_interactive_html(G, doc_names, clusters, languages, file_paths, repulsion_strength)
    
    analyze_clusters(doc_names, clusters, similarity_matrix, languages)
    
    print("\n✓ Mehrsprachige Analyse abgeschlossen!")

if __name__ == "__main__":
    pdf_folder = r"..."         # replace ... with your directory
    
    if os.path.exists(pdf_folder):
        main(
            folder_path=pdf_folder,
            similarity_threshold=0.2,  # Niedriger für mehrsprachig
            n_clusters=None,  # Automatisch
            multilingual=True,  # Mehrsprachig aktiviert
            repulsion_strength=3.5,  # 1.0-5.0: Höher = stärkere Abstoßung schwacher Kanten
            spring_k=14.0  # 3.0-15.0: Höher = mehr Gesamtabstand zwischen allen Knoten
        )
    else:
        print(f"⚠ Ordner nicht gefunden: {pdf_folder}")
        print("\nErstelle einen Ordner 'pdfs' mit deinen PDF-Dateien.")
        print("Das Skript unterstützt deutsche und englische Dokumente gemischt!")
        print("\nParameter-Tuning:")
        print("  repulsion_strength: 1.0 (schwach) - 5.0 (sehr stark)")

        print("  spring_k: 3.0 (kompakt) - 15.0 (sehr weit verteilt)")
