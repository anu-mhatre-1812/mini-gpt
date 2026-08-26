import json
import urllib.parse
import urllib.request

titles = [
    "महात्मा_गांधी", "मराठी_भाषा", "महाराष्ट्र", "पुणे", "मुंबई",
    "संगणक", "क्रिकेट", "शिवाजी_महाराज", "भारत", "विज्ञान",
    "गणित", "संगीत", "आरोग्य", "शिक्षण", "इतिहास",
    "तंत्रज्ञान", "प्राणी", "वृक्ष", "आकाशगंगा", "सूर्य",
    "नदी", "पर्वत", "खेळ", "चित्रपट", "साहित्य",
]

out = []
for t in titles:
    try:
        url = ("https://mr.wikipedia.org/w/api.php?action=query&prop=extracts"
               f"&explaintext=1&format=json&titles={urllib.parse.quote(t)}")
        req = urllib.request.Request(url, headers={"User-Agent": "MiniGPT/0.1 (training data)"})
        data = json.loads(urllib.request.urlopen(req, timeout=20).read())
        for page in data["query"]["pages"].values():
            ext = page.get("extract", "")
            if len(ext) > 500:
                out.append(ext)
                print(f"  + {t} ({len(ext):,} chars)")
    except Exception as e:
        print(f"  - skip {t}: {str(e)[:50]}")

text = "\n\n".join(out)
with open(r"C:\Users\ADMIN\Projects\mini-gpt\data\marathi.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"\nSaved {len(out)} articles, {len(text):,} chars total")
