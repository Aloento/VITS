""" from https://github.com/keithito/tacotron """
import re

import eng_to_ipa as ipa
from g2p_en import G2p
from unidecode import unidecode

from text.frontend import normalize_numbers

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

# Regular expression matching whitespace:
g2p = G2p()

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

# List of (ipa, ipa2) pairs
_ipa_to_ipa2 = [(re.compile('%s' % x[0]), x[1]) for x in [
  ('r', 'ɹ'),
  ('ʤ', 'dʒ'),
  ('ʧ', 'tʃ')
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def collapse_whitespace(text):
  return re.sub(r'\s+', ' ', text)


def mark_dark_l(text):
  return re.sub(r'l([^aeiouæɑɔəɛɪʊ ]*(?: |$))', lambda x: 'ɫ' + x.group(1), text)


def english_to_ipa(text):
  text = text.replace("-", " ")
  text = unidecode(text).lower()
  text = expand_abbreviations(text)
  text = normalize_numbers(text)

  phonemes = ipa.convert(text)
  phonemes = unrecognized_words_to_ipa(phonemes)
  phonemes = collapse_whitespace(phonemes)

  text = phonemes
  text = mark_dark_l(text)

  for regex, replacement in _ipa_to_ipa2:
    text = re.sub(regex, replacement, text)

  return text.replace('...', '…')


def convert_to_ipa(phones):
  eipa = ""
  symbols = {"a": "ə", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
             "aw": "aʊ", "ay": "aɪ", "ch": "ʧ", "dh": "ð", "eh": "ɛ", "er": "ər",
             "hh": "h", "ih": "ɪ", "jh": "ʤ", "ng": "ŋ", "ow": "oʊ", "oy": "ɔɪ",
             "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j"}

  for ph in phones:
    ph = ph.lower()

    try:
      if ph[-1] in "01234":
        eipa += symbols[ph[:-1]]
      else:
        eipa += symbols[ph]
    except:
      eipa += ph

  return eipa


def unrecognized_words_to_ipa(text):
  matches = re.findall(r'\s([\w|\']+\*)', text)

  for word in matches:
    ipa = convert_to_ipa(g2p(word))
    text = text.replace(word, ipa)

  matches = re.findall(r'^([\w|\']+\*)', text)

  for word in matches:
    ipa = convert_to_ipa(g2p(word))
    text = text.replace(word, ipa)

  return text
