import os
import pandas as pd
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

FILE_PATH = r"C:\Users\MattO\Documents\ESADE\ESADE\research\processed_earnings_calls\earnings_calls_combined_2018_current.csv"
OUTPUT_DIR = "raw_data"

GVKEY = 1004  # AAR Corp

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(FILE_PATH)
result = df[df["gvkey"] == GVKEY]

styles = getSampleStyleSheet()

for (gvkey, year, transcript_id), group in result.groupby(["gvkey", "Year", "transcriptId"]):
    filename = os.path.join(OUTPUT_DIR, f"{gvkey}_{year}_{transcript_id}.pdf")
    doc = SimpleDocTemplate(filename, pagesize=LETTER)
    content = [Paragraph(row["componentText"], styles["Normal"]) for _, row in group.iterrows()]
    doc.build(content)
    print(f"Saved: {filename}")
