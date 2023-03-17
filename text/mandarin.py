import re

import cn2an

# List of (Latin alphabet, bopomofo) pairs:
from text.paddle_zh import zh_to_bopomofo, pinyin_to_bopomofo

_latin_to_bopomofo = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
  ('a', 'ㄟˉ'),
  ('b', 'ㄅㄧˋ'),
  ('c', 'ㄙㄧˉ'),
  ('d', 'ㄉㄧˋ'),
  ('e', 'ㄧˋ'),
  ('f', 'ㄝˊㄈㄨˋ'),
  ('g', 'ㄐㄧˋ'),
  ('h', 'ㄝˇㄑㄩˋ'),
  ('i', 'ㄞˋ'),
  ('j', 'ㄐㄟˋ'),
  ('k', 'ㄎㄟˋ'),
  ('l', 'ㄝˊㄛˋ'),
  ('m', 'ㄝˊㄇㄨˋ'),
  ('n', 'ㄣˉ'),
  ('o', 'ㄡˉ'),
  ('p', 'ㄆㄧˉ'),
  ('q', 'ㄎㄧㄡˉ'),
  ('r', 'ㄚˋ'),
  ('s', 'ㄝˊㄙˋ'),
  ('t', 'ㄊㄧˋ'),
  ('u', 'ㄧㄡˉ'),
  ('v', 'ㄨㄧˉ'),
  ('w', 'ㄉㄚˋㄅㄨˋㄌㄧㄡˋ'),
  ('x', 'ㄝˉㄎㄨˋㄙˋ'),
  ('y', 'ㄨㄞˋ'),
  ('z', 'ㄗㄟˋ')
]]

# List of (bopomofo, ipa) pairs:
_bopomofo_to_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [
  ('ㄅㄛ', 'p⁼wo'),
  ('ㄆㄛ', 'pʰwo'),
  ('ㄇㄛ', 'mwo'),
  ('ㄈㄛ', 'fwo'),
  ('ㄅ', 'p⁼'),
  ('ㄆ', 'pʰ'),
  ('ㄇ', 'm'),
  ('ㄈ', 'f'),
  ('ㄉ', 't⁼'),
  ('ㄊ', 'tʰ'),
  ('ㄋ', 'n'),
  ('ㄌ', 'l'),
  ('ㄍ', 'k⁼'),
  ('ㄎ', 'kʰ'),
  ('ㄏ', 'x'),
  ('ㄐ', 'tʃ⁼'),
  ('ㄑ', 'tʃʰ'),
  ('ㄒ', 'ʃ'),
  ('ㄓ', 'ts`⁼'),
  ('ㄔ', 'ts`ʰ'),
  ('ㄕ', 's`'),
  ('ㄖ', 'ɹ`'),
  ('ㄗ', 'ts⁼'),
  ('ㄘ', 'tsʰ'),
  ('ㄙ', 's'),
  ('ㄚ', 'a'),
  ('ㄛ', 'o'),
  ('ㄜ', 'ə'),
  ('ㄝ', 'ɛ'),
  ('ㄞ', 'aɪ'),
  ('ㄟ', 'eɪ'),
  ('ㄠ', 'ɑʊ'),
  ('ㄡ', 'oʊ'),
  ('ㄧㄢ', 'jɛn'),
  ('ㄩㄢ', 'ɥæn'),
  ('ㄢ', 'an'),
  ('ㄧㄣ', 'in'),
  ('ㄩㄣ', 'ɥn'),
  ('ㄣ', 'ən'),
  ('ㄤ', 'ɑŋ'),
  ('ㄧㄥ', 'iŋ'),
  ('ㄨㄥ', 'ʊŋ'),
  ('ㄩㄥ', 'jʊŋ'),
  ('ㄥ', 'əŋ'),
  ('ㄦ', 'əɻ'),
  ('ㄧ', 'i'),
  ('ㄨ', 'u'),
  ('ㄩ', 'ɥ'),
  ('ˉ', '→'),
  ('ˊ', '↑'),
  ('ˇ', '↓↑'),
  ('ˋ', '↓'),
  ('˙', ''),
  ('，', ','),
  ('。', '.'),
  ('！', '!'),
  ('？', '?'),
  ('—', '-')
]]


def number_to_chinese(text):
  numbers = re.findall(r'\d+(?:\.?\d+)?', text)
  for number in numbers:
    text = text.replace(number, cn2an.an2cn(number), 1)
  return text


def latin_to_bopomofo(text):
  for regex, replacement in _latin_to_bopomofo:
    text = re.sub(regex, replacement, text)
  return text


def bopomofo_to_ipa(text):
  for regex, replacement in _bopomofo_to_ipa:
    text = re.sub(regex, replacement, text)
  return text


def chinese_to_ipa(text):
  text = number_to_chinese(text)
  text = zh_to_bopomofo(text)
  text = _clean_zh(text)
  return text


def pinyin_to_ipa(text):
  text = pinyin_to_bopomofo(text)
  text = _clean_zh(text)
  text = text.replace("%", " %").replace("$", " $")
  return text


def _clean_zh(text):
  text = latin_to_bopomofo(text)
  text = bopomofo_to_ipa(text)
  text = re.sub('i([aoe])', r'j\1', text)
  text = re.sub('u([aoəe])', r'w\1', text)
  text = re.sub('([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ`\2', text).replace('ɻ', 'ɹ`')
  text = re.sub('(s[⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ\2', text)
  return text
