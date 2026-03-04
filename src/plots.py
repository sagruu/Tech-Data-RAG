import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def bar_plot(data, x_column, y_column, title, xlabel, ylabel, xrotation=0, ha_xrotation='center', height_offset=50):
    """Creates a bar plot with value annotations on top of each bar."""
    plt.figure(figsize=(12, 8))
    bars = plt.bar(data[x_column], data[y_column])

    # Werte auf den Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + height_offset, f'{height}', ha='center', va='bottom', fontsize=8)

    plt.xticks(rotation=90)
    plt.xlabel(xlabel)
    plt.xticks(rotation=xrotation, ha=ha_xrotation)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_wordcloud(text, c_stopwords=None, column_name=None):
    """Generates and displays a word cloud from text, filtering out custom stopwords."""
    stopwords = set(STOPWORDS).union(c_stopwords)
    wordcloud = WordCloud(stopwords=stopwords, max_words=30, background_color="white").generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Die häufigsten 30 Wörter aus der Spalte {column_name}")
    plt.axis("off")
    plt.show()