# Bayn recording corpus

Scripts to record once and reuse: to compare transcription models on **identical audio** (`bayn eval-stt --corpus`), to test summaries on realistic transcripts, and as a reference when you test with a live mic. The same 4 topics in 3 versions: English, Arabic, and mixed (Arabic with the English key terms a teacher would really use). Names are fictional.

> Generated from [tests/data/corpus.json](../tests/data/corpus.json) by `python -m bayn.corpus render`. Edit the JSON, not this file.

## How to record

1. Record each script once with `.\bayn record <id>` - the script opens in your browser in large type, one sentence per line (put it next to the terminal). Press Enter in the terminal to start, and again when you finish reading.
2. Same microphone, same distance (about 20-30 cm), quiet room, for every recording, so models are compared on equal terms.
3. Read naturally, like you would in class: normal speed, small pauses at commas and full stops. Don't over-articulate.
4. If you stumble, just continue (that's realistic) - or press Ctrl+C and record it again.
5. Recordings are saved in data/recordings/ and never leave this computer (not committed to git).

```powershell
.\bayn record mx-pi            # opens the script in big type, Enter stops
.\bayn corpus show mx-pi       # just open the reading page (e.g. for live-mic tests)
.\bayn eval-stt --corpus all   # score every recorded script
```

Scripts are shown one sentence per line. In the mixed scripts the **bold** words are English: say them in English, the way a teacher would in class.

| ID | Version | Topic | ~Time |
|---|---|---|---|
| `en-atom` | English only | The atom and Rutherford's experiment (science lesson) | 40 s |
| `en-pi` | English only | Circles and pi (maths lesson) | 37 s |
| `en-plants` | English only | Photosynthesis (biology lesson) | 38 s |
| `en-incident` | English only | A student becomes unresponsive in class (the event Bayn is built for) | 36 s |
| `ar-atom` | Arabic only | The atom and Rutherford's experiment (science lesson) | 39 s |
| `ar-pi` | Arabic only | Circles and pi (maths lesson) | 39 s |
| `ar-plants` | Arabic only | Photosynthesis (biology lesson) | 39 s |
| `ar-incident` | Arabic only | A student becomes unresponsive in class (the event Bayn is built for) | 36 s |
| `mx-atom` | Mixed (Arabic with English key terms) | The atom and Rutherford's experiment (science lesson) | 36 s |
| `mx-pi` | Mixed (Arabic with English key terms) | Circles and pi (maths lesson) | 34 s |
| `mx-plants` | Mixed (Arabic with English key terms) | Photosynthesis (biology lesson) | 29 s |
| `mx-incident` | Mixed (Arabic with English key terms) | A student becomes unresponsive in class (the event Bayn is built for) | 32 s |

## English only

### `en-atom` · The atom and Rutherford's experiment (science lesson) (~40 s)

<p dir="ltr" lang="en">
Good morning, everyone.<br>
Today we&#x27;re talking about the atom.<br>
About a hundred years ago, a scientist named Ernest Rutherford fired tiny particles at a very thin sheet of gold foil.<br>
Most of the particles went straight through, but a few of them bounced back.<br>
Rutherford concluded that almost all of the atom&#x27;s mass is packed into a tiny centre called the nucleus.<br>
The nucleus contains protons, which have a positive charge, and the electrons move around it.<br>
So, here&#x27;s a question for you: if most of the atom is empty space, why can&#x27;t we walk through walls?
</p>

**Key terms that must survive transcription:** `Rutherford`, `gold foil`, `nucleus`, `protons`, `electrons`, `atom`

### `en-pi` · Circles and pi (maths lesson) (~37 s)

<p dir="ltr" lang="en">
Alright, take out your rulers.<br>
Measure the distance around this circle.<br>
That&#x27;s the circumference.<br>
Now measure straight across, through the middle.<br>
That&#x27;s the diameter.<br>
If you divide the circumference by the diameter, you always get the same number, about three point one four.<br>
We call this number pi.<br>
It doesn&#x27;t matter if the circle is as small as a coin or as big as a wheel.<br>
So the formula is: the circumference equals pi times the diameter.<br>
For homework, find three round objects at home and test it yourself.
</p>

**Key terms that must survive transcription:** `circumference`, `diameter`, `pi`, `three point one four`, `formula`

### `en-plants` · Photosynthesis (biology lesson) (~38 s)

<p dir="ltr" lang="en">
Look at the plant on the windowsill.<br>
Plants make their own food through a process called photosynthesis.<br>
The leaves take in carbon dioxide from the air, and the roots take in water from the soil.<br>
A green substance in the leaves, called chlorophyll, captures energy from sunlight.<br>
Using that energy, the plant turns water and carbon dioxide into sugar, and it releases oxygen.<br>
That oxygen is what we breathe.<br>
Quick question: what do you think happens to a plant if we keep it in a dark cupboard for a week?
</p>

**Key terms that must survive transcription:** `photosynthesis`, `carbon dioxide`, `chlorophyll`, `oxygen`, `sugar`

### `en-incident` · A student becomes unresponsive in class (the event Bayn is built for) (~36 s)

<p dir="ltr" lang="en">
It&#x27;s ten twenty.<br>
Sara stopped answering in the middle of the reading.<br>
She&#x27;s staring at the board and she&#x27;s not responding to her name.<br>
Her lips are moving, like small chewing movements, and her left hand is fidgeting with her pencil.<br>
I&#x27;m timing it now.<br>
About forty seconds have passed.<br>
She didn&#x27;t fall, she&#x27;s still sitting in her chair.<br>
Okay, she&#x27;s blinking now and looking around.<br>
She says she feels tired and she doesn&#x27;t remember the last minute.<br>
Ahmed, please go and call the school nurse.
</p>

**Key terms that must survive transcription:** `ten twenty`, `Sara`, `staring`, `chewing`, `forty seconds`, `nurse`

*Summary test: a good summary reports what was observed (time, staring, chewing movements, about 40 s, didn't fall, tired, nurse called) and does **not** diagnose.*


## Arabic only

### `ar-atom` · The atom and Rutherford's experiment (science lesson) (~39 s)

<p dir="rtl" lang="ar">
صباح الخير يا شباب.<br>
اليوم بنتكلم عن الذرة.<br>
قبل حوالي مئة سنة، قام عالم اسمه إرنست رذرفورد بإطلاق جسيمات صغيرة جداً على صفيحة رقيقة من الذهب.<br>
معظم الجسيمات عبرت الصفيحة مباشرة، لكن بعضها ارتد للخلف.<br>
واستنتج رذرفورد أن معظم كتلة الذرة موجودة في مركز صغير جداً اسمه النواة.<br>
النواة فيها بروتونات شحنتها موجبة، والإلكترونات تتحرك حولها.<br>
طيب، سؤال لكم: إذا كان أغلب الذرة فراغ، ليش ما نقدر نمشي من خلال الجدار؟
</p>

**Key terms that must survive transcription:** <bdi>`رذرفورد`</bdi>, <bdi>`الذهب`</bdi>, <bdi>`النواة`</bdi>, <bdi>`بروتونات`</bdi>, <bdi>`الإلكترونات`</bdi>, <bdi>`الذرة`</bdi>

### `ar-pi` · Circles and pi (maths lesson) (~39 s)

<p dir="rtl" lang="ar">
طيب، طلّعوا المساطر.<br>
قيسوا المسافة حول هذه الدائرة، وهذا اسمه المحيط.<br>
الحين قيسوا المسافة من طرف لطرف مروراً بالمركز، وهذا اسمه القطر.<br>
إذا قسمنا المحيط على القطر نحصل دائماً على نفس العدد، تقريباً ثلاثة فاصلة واحد أربعة.<br>
نسمي هذا العدد ط.<br>
ما يفرق إذا كانت الدائرة صغيرة مثل العملة أو كبيرة مثل العجلة.<br>
إذن القانون: المحيط يساوي ط ضرب القطر.<br>
والواجب: دوروا على ثلاث أشياء دائرية في البيت وجربوا بأنفسكم.
</p>

**Key terms that must survive transcription:** <bdi>`المحيط`</bdi>, <bdi>`القطر`</bdi>, <bdi>`ط`</bdi>, <bdi>`ثلاثة فاصلة واحد أربعة`</bdi>, <bdi>`القانون`</bdi>

### `ar-plants` · Photosynthesis (biology lesson) (~39 s)

<p dir="rtl" lang="ar">
شوفوا النبتة اللي على الشباك.<br>
النباتات تصنع غذاءها بنفسها من خلال عملية اسمها البناء الضوئي.<br>
الأوراق تمتص ثاني أكسيد الكربون من الهواء، والجذور تمتص الماء من التربة.<br>
فيه مادة خضراء في الأوراق اسمها الكلوروفيل، تلتقط الطاقة من ضوء الشمس.<br>
وباستخدام هذه الطاقة يحوّل النبات الماء وثاني أكسيد الكربون إلى سكر، ويطلق الأكسجين.<br>
وهذا الأكسجين هو اللي نتنفسه.<br>
سؤال سريع: إيش تتوقعون يصير للنبتة لو حطيناها في دولاب مظلم لمدة أسبوع؟
</p>

**Key terms that must survive transcription:** <bdi>`البناء الضوئي`</bdi>, <bdi>`ثاني أكسيد الكربون`</bdi>, <bdi>`الكلوروفيل`</bdi>, <bdi>`الأكسجين`</bdi>, <bdi>`سكر`</bdi>

### `ar-incident` · A student becomes unresponsive in class (the event Bayn is built for) (~36 s)

<p dir="rtl" lang="ar">
الساعة الآن عشرة وعشرين.<br>
سارة توقفت عن الرد في نص القراءة.<br>
هي تحدّق في السبورة وما ترد لما أناديها باسمها.<br>
شفايفها تتحرك حركات صغيرة كأنها تمضغ، ويدها اليسرى تعبث بالقلم.<br>
أنا أحسب الوقت الآن.<br>
مرت تقريباً أربعين ثانية.<br>
ما طاحت، وهي لسا جالسة على الكرسي.<br>
طيب، الحين صارت ترمش وتطالع حولها.<br>
تقول إنها تعبانة وما تتذكر آخر دقيقة.<br>
أحمد، لو سمحت روح نادِ ممرضة المدرسة.
</p>

**Key terms that must survive transcription:** <bdi>`عشرة وعشرين`</bdi>, <bdi>`سارة`</bdi>, <bdi>`تحدّق`</bdi>, <bdi>`تمضغ`</bdi>, <bdi>`أربعين ثانية`</bdi>, <bdi>`ممرضة`</bdi>

*Summary test: a good summary reports what was observed (time, staring, chewing movements, about 40 s, didn't fall, tired, nurse called) and does **not** diagnose.*


## Mixed (Arabic with English key terms)

### `mx-atom` · The atom and Rutherford's experiment (science lesson) (~36 s)

<p dir="rtl" lang="ar">
صباح الخير يا شباب.<br>
اليوم درسنا عن الـ <b dir="ltr">atom</b>.<br>
قبل حوالي مئة سنة، العالم <b dir="ltr">Rutherford</b> أطلق جسيمات صغيرة على صفيحة رقيقة من الـ <b dir="ltr">gold foil</b>.<br>
معظم الجسيمات عبرت مباشرة، بس بعضها ارتد.<br>
فاستنتج <b dir="ltr">Rutherford</b> إن أغلب كتلة الذرة موجودة في مركز صغير اسمه الـ <b dir="ltr">nucleus</b>.<br>
الـ <b dir="ltr">nucleus</b> فيه <b dir="ltr">protons</b> شحنتها موجبة، والـ <b dir="ltr">electrons</b> تتحرك حوله.<br>
طيب، سؤال: إذا أغلب الذرة فراغ، ليش ما نقدر نمشي من خلال الجدار؟
</p>

**Key terms that must survive transcription:** <bdi>`atom`</bdi>, <bdi>`Rutherford`</bdi>, <bdi>`gold foil`</bdi>, <bdi>`nucleus`</bdi>, <bdi>`protons`</bdi>, <bdi>`electrons`</bdi>

### `mx-pi` · Circles and pi (maths lesson) (~34 s)

<p dir="rtl" lang="ar">
طيب، طلّعوا المساطر.<br>
قيسوا المسافة حول الدائرة، وهذا اسمه الـ <b dir="ltr">circumference</b>.<br>
وبعدين قيسوا من طرف لطرف عبر المركز، وهذا الـ <b dir="ltr">diameter</b>.<br>
إذا قسمنا الـ <b dir="ltr">circumference</b> على الـ <b dir="ltr">diameter</b> نطلع دائماً بنفس الرقم، تقريباً <b dir="ltr">three point one four</b>، ونسميه <b dir="ltr">pi</b>.<br>
ما يفرق إذا الدائرة صغيرة أو كبيرة.<br>
إذن الـ <b dir="ltr">formula</b>: الـ <b dir="ltr">circumference</b> يساوي <b dir="ltr">pi</b> ضرب الـ <b dir="ltr">diameter</b>.<br>
والواجب: جربوها على ثلاث أشياء دائرية في البيت.
</p>

**Key terms that must survive transcription:** <bdi>`circumference`</bdi>, <bdi>`diameter`</bdi>, <bdi>`three point one four`</bdi>, <bdi>`pi`</bdi>, <bdi>`formula`</bdi>

### `mx-plants` · Photosynthesis (biology lesson) (~29 s)

<p dir="rtl" lang="ar">
شوفوا النبتة اللي على الشباك.<br>
النبات يصنع غذاءه بعملية اسمها <b dir="ltr">photosynthesis</b>.<br>
الأوراق تمتص الـ <b dir="ltr">carbon dioxide</b> من الهواء، والجذور تمتص الماء.<br>
فيه مادة خضراء اسمها <b dir="ltr">chlorophyll</b> تلتقط طاقة الشمس.<br>
وبهذي الطاقة يحول الماء والـ <b dir="ltr">carbon dioxide</b> إلى <b dir="ltr">sugar</b> ويطلق الـ <b dir="ltr">oxygen</b>، وهو اللي نتنفسه.<br>
سؤال سريع: إيش يصير للنبتة لو حطيناها في دولاب مظلم أسبوع كامل؟
</p>

**Key terms that must survive transcription:** <bdi>`photosynthesis`</bdi>, <bdi>`carbon dioxide`</bdi>, <bdi>`chlorophyll`</bdi>, <bdi>`sugar`</bdi>, <bdi>`oxygen`</bdi>

### `mx-incident` · A student becomes unresponsive in class (the event Bayn is built for) (~32 s)

<p dir="rtl" lang="ar">
الساعة الحين عشرة وعشرين.<br>
سارة وقفت عن الرد في نص الـ <b dir="ltr">reading</b>.<br>
تحدق في الـ <b dir="ltr">board</b> وما ترد لما أناديها.<br>
شفايفها تتحرك كأنها تمضغ، ويدها اليسرى تلعب بالقلم.<br>
أنا أحسب الوقت الآن، تقريباً <b dir="ltr">forty seconds</b>.<br>
ما طاحت، لسا جالسة على الكرسي.<br>
طيب، الحين ترمش وتطالع حولها، وتقول إنها <b dir="ltr">tired</b> وما تتذكر آخر دقيقة.<br>
أحمد، روح نادِ الـ <b dir="ltr">school nurse</b> لو سمحت.
</p>

**Key terms that must survive transcription:** <bdi>`reading`</bdi>, <bdi>`board`</bdi>, <bdi>`forty seconds`</bdi>, <bdi>`tired`</bdi>, <bdi>`school nurse`</bdi>

*Summary test: a good summary reports what was observed (time, staring, chewing movements, about 40 s, didn't fall, tired, nurse called) and does **not** diagnose.*

