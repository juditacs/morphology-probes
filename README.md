This repository contains the data, the source code and the result tables for the paper **Morphosyntactic Probing of Multilingual BERT Models**.

# Dataset

The dataset contains 247 morphosyntactic probing tasks across 42 languages.
Each task is a triplet of <language, POS, morphosyntactic tag> such as <English, VERB, Tense>.
The task is to predict the morphosyntactic tag of a token given its sentence context.

**Languages**: Afrikaans, Albanian, Arabic, Armenian, Basque, Belarusian, Bulgarian, Catalan, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Irish, Italian, Latin, Latvian, Lithuanian, Norwegian Bokmal, Norwegian Nynorsk, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swedish, Turkish, Ukrainian, Urdu.

**Tasks**:
* <ADJ, case>
* <ADJ, gender>
* <ADJ, number>
* <ADJ, tense>
* <NOUN, case>
* <NOUN, gender>
* <NOUN, number>
* <PROPN, case>
* <PROPN, gender>
* <PROPN, number>
* <VERB, case>
* <VERB, gender>
* <VERB, number>
* <VERB, tense>

![Task availability by language](/fig/heatmap_task_language_availability.png)

# Running probing experiments

The source code for running the experiments is available in the [probing](https://github.com/juditacs/probing) package.
