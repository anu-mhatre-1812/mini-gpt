"""Build the multi-domain training corpus (teacher: ox-alpha).

Domains: Marathi (clean) + Hindi + English + Maths patterns + Code.
Combined with the Wikipedia Marathi data into data/combined.txt
"""

from pathlib import Path

DATA = Path(__file__).parent

MARATHI_CLEAN = """मराठी भाषा भारतातील महाराष्ट्र राज्याची अधिकृत भाषा आहे. ही भाषा देवनागरी लिपीत लिहिली जाते. मराठी भाषेचा इतिहास अतिशय जुना आहे. संत ज्ञानेश्वरांनी ज्ञानेश्वरी ग्रंथ लिहिला. त्यानंतर संत तुकाराम आणि संत नामदेव यांनी अभंग रचले. छत्रपती शिवाजी महाराजांनी स्वराज्य स्थापन केले. मराठी साहित्य संमेलने दरवर्षी भरतात. पुणे शहर म्हणजे मराठी साहित्याचे माहेरघर. लोकमान्य तिळक म्हणाले, स्वराज्य हा माझा जन्मसिद्ध हक्क आहे. मराठी भाषेत नाटक, कविता, कथा आणि कादंबऱ्या लिहिल्या जातात.

महाराष्ट्रात मुंबई ही राजधानी आहे. मुंबई हे भारताचे आर्थिक महानगर आहे. येथे बॉलिवूड चित्रपटसृष्टी आहे. नाशिक शहर द्राक्षांसाठी प्रसिद्ध आहे. नागपूर हे संत्र्यांसाठी प्रसिद्ध आहे. कोकणचा किनारा अतिशय सुंदर आहे. सह्याद्री पर्वतात जंगले आहेत. गोदावरी आणि कृष्णा या मोठ्या नद्या आहेत. शेतकरी भात, ऊस आणि ज्वारी पिकवतात. गणेशोत्सव हा सर्वात मोठा सण आहे. दिवाळी आणि होळी हे सण सर्वजण साजरे करतात.

विज्ञानामुळे जग बदलले आहे. सूर्य पृथ्वीभोवती फिरत नाही, तर पृथ्वी सूर्याभोवती फिरते. पाणी ही H2O ची बनलेली असते. वनस्पती प्रकाशापासून अन्न बनवतात. या प्रक्रियेला प्रकाशसंश्लेषण म्हणतात. माणसाला श्वास घेण्यासाठी ऑक्सिजन लागतो. वृक्ष ऑक्सिजन देतात. म्हणून झाडे लावावीत. पाऊस पडल्यास शेती भरभराटते. वैद्यकीय शास्त्राने अनेक रोगांवर औषधे शोधली आहेत."""

HINDI = """भारत एक विशाल देश है। यहाँ कई भाषाएँ बोली जाती हैं। हिंदी भारत की राष्ट्रभाषा है। दिल्ली भारत की राजधानी है। ताजमहल आगरा में स्थित है। यह विश्व का सबसे सुंदर स्मारक है। गंगा नदी हिमालय से निकलती है। यह नदी बहुत पवित्र मानी जाती है। किसान खेत में काम करते हैं। वे गेहूँ, चावल और मक्का उगाते हैं। बच्चे स्कूल जाते हैं। शिक्षक उन्हें पढ़ाते हैं। शिक्षा जीवन की नींव है। मेहनत का फल मीठा होता है। समय सबसे मूल्यवान धन है। स्वस्थ शरीर में स्वस्थ मन रहता है।

विज्ञान ने जीवन बदल दिया है। बिजली से घर रोशन होते हैं। ट्रेन, बस और हवाई जहाज यात्रा आसान करते हैं। मोबाइल फोन से दूर बैठे लोगों से बात हो जाती है। इंटरनेट से दुनिया की हर जानकारी मिल जाती है। कंप्यूटर हर कार्यालय में है। वैज्ञानिक नए नए आविष्कार कर रहे हैं। चंद्रमा और मंगल पर भारत ने मिशन भेजे हैं। हमें प्रकृति की रक्षा करनी चाहिए। पेड़ लगाओ, जीवन बचाओ।"""

ENGLISH = """The sun rises in the east and sets in the west. Water flows downhill and finds its way to the sea. Trees give us oxygen, fruits and shade. Birds build nests and sing in the morning. Children go to school to learn reading and writing. Knowledge is the greatest treasure a person can own. Practice makes a person perfect. Hard work always pays off in the end. A journey of a thousand miles begins with a single step.

Machine learning is a branch of artificial intelligence. A model learns patterns from data instead of following fixed rules. Training means adjusting weights to reduce error. A neural network has layers of connected neurons. Attention lets a model focus on important parts of the input. Large language models predict the next token in a sequence. They are trained on huge amounts of text. The transformer architecture changed everything in modern AI. Data quality matters more than model size. Always evaluate your model on data it has never seen."""

MATHS = """1 + 1 = 2
1 + 2 = 3
2 + 2 = 4
2 + 3 = 5
3 + 3 = 6
4 + 4 = 8
5 + 5 = 10
10 + 10 = 20
25 + 25 = 50
50 + 50 = 100
2 x 1 = 2
2 x 2 = 4
2 x 3 = 6
3 x 3 = 9
4 x 4 = 16
5 x 5 = 25
6 x 6 = 36
7 x 7 = 49
8 x 8 = 64
9 x 9 = 81
10 x 10 = 100
12 x 12 = 144
10 / 2 = 5
100 / 4 = 25
100 - 25 = 75
a + b = c
a x a = a squared
the square root of 16 is 4
the square root of 81 is 9
pi is approximately 3.14
the sum of angles in a triangle is 180 degrees
a right angle is 90 degrees
2, 4, 6, 8, 10 are even numbers
1, 3, 5, 7, 9 are odd numbers
2, 3, 5, 7, 11 are prime numbers
1, 1, 2, 3, 5, 8, 13 is the fibonacci sequence
each number is the sum of the previous two numbers
x + 5 = 10 therefore x = 5
2x = 10 therefore x = 5
the area of a square is side times side
the area of a circle is pi times r squared"""

CODE = """# python basics
x = 5
y = 10
print(x + y)
for i in range(10):
    print(i)
if x > y:
    print("x is greater")
else:
    print("y is greater")

def add(a, b):
    return a + b

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
print(total)

# c basics
#include <stdio.h>
int main() {
    printf("hello world");
    return 0;
}

for (int i = 0; i < 10; i++) {
    printf("%d", i);
}

// binary search
int low = 0;
int high = n - 1;
while (low <= high) {
    int mid = (low + high) / 2;
    if (arr[mid] == target) return mid;
    if (arr[mid] < target) low = mid + 1;
    else high = mid - 1;
}

// sorting
void sort(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (arr[i] > arr[j]) {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
}"""


def main():
    files = {
        "marathi_clean.txt": MARATHI_CLEAN,
        "hindi.txt": HINDI,
        "english.txt": ENGLISH,
        "maths.txt": MATHS,
        "code.txt": CODE,
    }
    parts = []
    for name, content in files.items():
        (DATA / name).write_text(content.strip(), encoding="utf-8")
        parts.append(content.strip())
        print(f"  {name}: {len(content):,} chars")

    wiki = DATA / "marathi.txt"
    if wiki.exists():
        parts.append(wiki.read_text(encoding="utf-8"))
        print(f"  marathi.txt (wikipedia): {wiki.stat().st_size:,} chars")

    combined = "\n\n".join(parts)
    (DATA / "combined.txt").write_text(combined, encoding="utf-8")
    print(f"\ncombined.txt: {len(combined):,} chars total")


if __name__ == "__main__":
    main()
