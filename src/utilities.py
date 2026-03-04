import pandas as pd
import re
from urllib.parse import urlparse, parse_qs
from IPython.display import Markdown, display
import nbformat
import os


# Funktion Berechnung % Anteils von fehlenden Werten und Nullen
def missing_value_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistics on missing (NaN/None) and zero values for each column in a DataFrame.

    For each column in the input DataFrame, this function calculates:
      - The percentage of missing values (NaN or None)
      - The percentage of zero values (only for numeric columns)

    The results are returned in a transposed DataFrame for easier readability,
    with one row per column of the input DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        pandas.DataFrame: A transposed DataFrame where each row corresponds to
        a column from the input, with the following statistics:
            - "NaN/None (%)": Percentage of missing values.
            - "0-Werte (%)": Percentage of zero values (numeric columns only).
    """
    total_rows = len(df)

    stats = {}
    for col in df.columns:
        num_missing = df[col].isna().sum()  # Anzahl NaN/None
        num_zeros = (df[col] == 0).sum() if df[col].dtype in ["int64", "float64"] else 0  # Anzahl 0-Werte

        stats[col] = {
            "NaN/None (%)": round((num_missing / total_rows) * 100, 2),
            "0-Werte (%)": round((num_zeros / total_rows) * 100, 2)
        }

    return pd.DataFrame(stats).T  # Transponiert für Lesbarkeit

def shorten_text(val: str, max_lines: int = 5, max_chars_per_line: int = 50) -> str:
    """
    Shortens a given text string to a maximum number of lines and characters per line.

    This function splits the input string into lines, and each line is further
    truncated into chunks with at most `max_chars_per_line` characters. It stops
    collecting lines once the total reaches `max_lines`. If the limit is exceeded,
    an ellipsis ("...") is appended at the end.

    Parameters:
        val (str): The input string to be shortened.
        max_lines (int, optional): Maximum number of lines allowed in the output. Defaults to 5.
        max_chars_per_line (int, optional): Maximum number of characters per line. Defaults to 50.

    Returns:
        str: A shortened version of the input string respecting the specified limits.
             Returns the input unchanged if it is not a string.
    """
    if not isinstance(val, str):
        return val
    lines = val.splitlines()
    shortened_lines = []
    for line in lines:
        # Aufteilen in Stücke von max_chars_per_line
        for i in range(0, len(line), max_chars_per_line):
            shortened_lines.append(line[i:i+max_chars_per_line])
            if len(shortened_lines) >= max_lines:
                return '\n'.join(shortened_lines) + '\n...'
    return '\n'.join(shortened_lines)

def styled_text(val: str, max_lines: int = 5, max_chars_per_line: int = 50, max_width: int = 300) -> pd.DataFrame:
    """
    Applies formatting and styling to a DataFrame of text values for improved display.

    This function shortens each cell's text using the `shorten_text` function to limit
    the number of lines and characters per line. It then applies HTML/CSS styling to:
      - Preserve line breaks (`pre-wrap`)
      - Restrict maximum cell width
      - Use a monospace font for better readability

    Parameters:
        val (pandas.DataFrame): A DataFrame containing text values to format.
        max_lines (int, optional): Maximum number of lines per cell. Defaults to 5.
        max_chars_per_line (int, optional): Maximum number of characters per line. Defaults to 50.
        max_width (int, optional): Maximum width (in pixels) for each cell. Defaults to 300.

    Returns:
        pandas.io.formats.style.Styler: A styled DataFrame suitable for rendering in Jupyter or web.
    """
    styled = (val).map(shorten_text, max_lines=max_lines, max_chars_per_line=max_chars_per_line).style.set_properties(**{
            'white-space': 'pre-wrap',    # \n sichtbar
            'max-width': f"{max_width}px",         # Zellbreite beschränken
            'font-family': 'monospace',   # bessere Darstellung
        })
    return styled

def print_full_text(df: pd.DataFrame, max_width: int = 300) -> pd.DataFrame:
    """
    Displays the full, untruncated text content of a DataFrame with styled formatting.

    This function applies CSS styling to a DataFrame to enhance readability of
    long text entries by:
      - Preserving line breaks (`pre-wrap`)
      - Restricting the maximum display width of cells
      - Enabling word-breaking to prevent horizontal scrolling

    Parameters:
        df (pandas.DataFrame): The DataFrame containing text values to display.
        max_width (int, optional): Maximum width (in pixels) for each cell. Defaults to 300.

    Returns:
        pandas.io.formats.style.Styler: A styled DataFrame for display with full text shown.
    """
    return df.style.set_properties(**{
        'white-space': 'pre-wrap',     # erlaubt Zeilenumbrüche
        'max-width': f"{max_width}px", # Zellbreite beschränken
        'word-wrap': 'break-word',     # bricht bei Wortgrenzen
        })


class TextCleaner:
    """
    A utility class for cleaning and sanitizing text content, particularly suited for 
    removing boilerplate, cookie banners, legal disclaimers, navigation elements, 
    and other common web text clutter.

    Cleaning levels:
        - "safe": Removes only clearly ignorable text (e.g., cookie banners).
        - "risky": Also removes likely boilerplate and legal text.
        - "dangerous": Aggressively removes generic words and phrases, potentially overzealous.
    """
    def __init__(self, level: str = "safe") -> None:
        """
        Initializes the TextCleaner with a specified aggressiveness level.

        Parameters:
            level (str): One of "safe", "risky", or "dangerous", determining
                         how aggressively text will be cleaned.
        """
        self.level = level.lower()
        self.safe_patterns = [
            # Cookie & Datenschutz
            r"this website uses cookies.*?\.",
            r"the cookie settings on this website.*?\.",
            r"accept all cookies.*?\.",
            r"cookie settings.*?\.",
            r"privacy (policy|statement).*?\.",
            r"data protection policy.*?\.",
            r"view our privacy policy.*?\.",
            r"terms of (use|service).*?\.",
            r"consent.*?(data|cookies|tracking).*?\.",

            # Login / Account / Captcha
            r"sign (in|up) to.*?\.",
            r"log (in|out) to.*?\.",
            r"register to.*?\.",
            r"create an account.*?\.",
            r"remember me.*?\.",
            r"reset password.*?",
            r"click here to.*?(sign in|continue|reset).*?",
            r"i am not a robot",
            r"please enable javascript.*?\.",

            # E-Mail / Login-Felder
            r"(your )?email (address)?[\s:]*",
            r"(your )?password[\s:]*",

            # Navigations-/Feature-Labels
            r"\bfeatured content\b",
            r"(news|data) services",
            r"client support",
            r"daily gpi",
            r"product suite",
            r"hub and flow",
            r"\btrending news\b",
            r"download latest pdf edition",
            r"listen to.*?(ngi|podcast).*?\.",
            r"(markets|related topics).*?$",

            # Rechtliches
            r"copyright( ©)?[^.]*",
            r"all rights reserved\.?",
            r"issn\s+\d+",

            # URLs
            r"http\S+|www\S+",

            # E-Mail-Adressen (generisch)
            r"email\s+[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",

            # Social-Media-Handles
            r"@?\s*mveazey\s*ngi",
        ]


        self.risky_patterns = [
            # Footer-Texte mit Eigentümerangaben
            r"recharge is part of.*?nhst global publications.*?\.",
            r"recharge is part of dn media group.*?\.",
            r"dn media group.*?responsible for.*?\.",
            r"we use your data to.*?\.",

            # Hinweise auf Datenschutz oder Werbung
            r"we use cookies.*?use of cookies.*?(more info\.?)?",
            r"advertise with.*?\.",
            r"the content produced by.*?subsidiaries\.",
            r"opinions and comments.*?subsidiaries\.",

            # Allgemeine Navigations- oder Infohinweise
            r"read more at.*?\.",
            r"read the full article at.*?\.",
            r"for more information.*?\.",
            r"get our daily (emails|updates).*?\.",
            r"(receive|sign up for) (the )?daily (emails|news).*?\.",
            r"follow (the )?(topics|.*?content).*?daily (emails|updates).*?\.",
            r"subscribe (to .*?)?newsletter.*?\.",
        ]

        self.dangerous_patterns = [
            # Einzelwörter oder kurze Begriffe
            r"\bcontinue\b",
            r"\breset\b",
            r"\bnews\b",
            r"\bmarkets\b",

            # Generische Slogans oder Navigationsbegriffe
            r"\bdaily gas price index\b",
            r"\bdownload latest pdf edition\b",
            r"\btrending news\b",
            r"\bclient support\b",
            r"\brelated topics\b",

            # Sehr allgemeine Hinweise
            r"more info\.?",
            r"read more\b",
            r"read more here\b",
        ]

        # Marker, ab denen das Textende abgeschnitten wird
        self.end_markers = [
            "this content is protected",
            "if you want to cooperate",
            "please be mindful",
            "required fields are marked",
            "save my name",
            "by submitting this form",
            "your personal data",
            "data protection policy",
            "view our privacy policy",
            "cookie settings",
            "accept all cookies",
            "the cookie settings on this website",
        ]

    def get_patterns(self) -> list[str]:
        """
        Returns the list of regex patterns based on the selected cleaning level.

        Returns:
            list[str]: A list of regular expressions to apply for text cleaning.
        """
        if self.level == "safe":
            return self.safe_patterns
        elif self.level == "risky":
            return self.risky_patterns + self.safe_patterns
        elif self.level == "dangerous":
            return self.dangerous_patterns + self.risky_patterns + self.safe_patterns
        else:
            return self.safe_patterns
        
    def truncate_boilerplate_tail(self, text: str) -> str:
        """
        Truncates the input text at predefined end markers commonly found in footers or boilerplate sections.

        Parameters:
            text (str): The input text string.

        Returns:
            str: Truncated text before the first occurrence of a known end marker.
        """
        if not isinstance(text, str):
            return ""
        for marker in self.end_markers:
            idx =text.lower().find(marker)
            if idx != -1:
                return text[:idx].strip()
        return text.strip()
        
    def clean_text(self, text: str) -> str:
        """
        Cleans a single text string by removing unwanted boilerplate using regex patterns,
        normalizing characters, and truncating known footer content.

        Parameters:
            text (str): The input string to clean.

        Returns:
            str: A cleaned and normalized version of the input string.
        """
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r"http\S+|www\S+", "", text) # URLs entfernen (nochmal zur Sicherheit :D))
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"') # Typographische Zeichen vereinheitlichen

        for pattern in self.get_patterns():
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Zeichen vereinheitlichen und säubern
        text = re.sub(r"[-_/]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß,.!?\'\" ]", "", text)
        text = self.truncate_boilerplate_tail(text)

        return text.strip()
    
    def is_meaningful(self, text: str) -> bool:
        """
        Determines whether a cleaned text string contains meaningful content.

        Parameters:
            text (str): A string to evaluate.

        Returns:
            bool: True if the string has substantial non-meta content, False otherwise.
        """
        if not isinstance(text, str):
            return False
        cleaned = text.strip()
        has_substance = bool(re.search(r"[a-zA-Z0-9äöüÄÖÜß]", cleaned))
        is_not_meta = not re.search(r"\b(ngi|issn|[0-9]{4})\b", cleaned, flags=re.IGNORECASE)
        return has_substance and is_not_meta

    def clean_text_column(self, df: pd.DataFrame, column: str, new_column: str = None, keep_all_rows: bool = True) -> pd.DataFrame:
        """
        Applies text cleaning to a column in a DataFrame.

        Parameters:
            df (pandas.DataFrame): The input DataFrame.
            column (str): The name of the column to clean.
            new_column (str, optional): Name for the cleaned column. If None, the original column is overwritten.
            keep_all_rows (bool, optional): If False, drops rows where cleaned text is not meaningful. Defaults to True.

        Returns:
            pandas.DataFrame: The DataFrame with the cleaned text column added or updated.
        """
        target_column = new_column if new_column else column
        df[target_column] = df[column].apply(self.clean_text)

        if not keep_all_rows:
            df = df[df[target_column].apply(self.is_meaningful)]

        return df
    
# Automatisches Inhaltsverzeichnis für Jupyter Notebooks
def generate_toc(nb_path: str = None) -> None:
    """
    Generates and displays a Markdown-based table of contents (TOC) for a Jupyter Notebook.

    This function parses the notebook file, extracts Markdown headers (`#` to `######`),
    and constructs a navigable TOC using anchor links compatible with Jupyter's rendering.

    Parameters:
        nb_path (str, optional): Path to the `.ipynb` file. If None, attempts to detect
                                 the current notebook path using `ipynbname`.

    Side Effects:
        Displays the table of contents inline using IPython's Markdown display.

    Notes:
        - Requires the `ipynbname` package for automatic path detection.
        - Anchors are created by lowercasing header text, stripping punctuation,
          and replacing spaces with hyphens.
        - Headers of level 2 (`##`) and deeper are indented accordingly.

    Example:
        >>> generate_toc()  # Automatically detects current notebook
        >>> generate_toc("notebooks/example.ipynb")  # For a specific file
    """
    
    if nb_path is None:
        try:
            nb_path = ipynbname.path()
        except Exception:
            display(Markdown("*Fehler: Notebook-Dateipfad konnte nicht erkannt werden.*"))
            return
    
    with open(nb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    toc_lines = ["#Inhaltsverzeichnis\n"]
    
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            for line in cell.source.splitlines():
                match = re.match(r'^(#{1,6})\s+(.*)', line)
                if match:
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    anchor = re.sub(r'[^\w\s-]', '', title).replace(' ', '-')
                    anchor = anchor.lower()
                    indent = '  ' * (level - 1)
                    toc_lines.append(f"{indent}- [{title}](#{anchor})")

    toc_md = "\n".join(toc_lines)
    display(Markdown(toc_md))