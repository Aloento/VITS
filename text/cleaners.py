import re

from text import cleaned_text_to_sequence
from text.english import english_to_ipa
from text.japanese import japanese_to_ipa
from text.mandarin import chinese_to_ipa, pinyin_to_ipa
from text.symbols import symbols


def str_replace(data):
  zh_tab = [";", ":", "\"", "'"]
  eng_tab = [".", ",", ' ', " "]

  for index in range(len(zh_tab)):
    if zh_tab[index] in data:
      data = data.replace(zh_tab[index], eng_tab[index])

  return data


def clean_text(text):
  cleaned_text, lang_seq = cje_cleaner(text)
  cleaned_text = str_replace(cleaned_text)
  cleaned_text, lang_seq = remove_invalid_text(cleaned_text, lang_seq)

  return cleaned_text, lang_seq


def text_to_sequence(text):
  cleaned_text, lang_seq = clean_text(text)
  return cleaned_text_to_sequence(cleaned_text), lang_seq


lang_map = {
  "ZH": 0,
  "JA": 1,
  "EN": 3,
  "P": 0,
  "other": 5
}


def cje_cleaner(text: str):
  text = str_replace(text).replace("\"", '')
  # find all text blocks enclosed in [JA], [ZH], [EN], [P]
  original_text = text
  blocks = re.finditer(r'\[(JA|ZH|EN|P)\](.*?)\[\1\]', text)
  cleaned_text = ""
  lang_seq = []
  last_end = 0

  for block in blocks:
    start, end = block.span()
    # insert text not enclosed in any blocks
    ipa = original_text[last_end:start]
    lang_seq += [lang_map["other"] for i in ipa]
    cleaned_text += ipa
    last_end = end
    language = block.group(1)
    text = block.group(2)

    if language == 'P':
      ipa = pinyin_to_ipa(text)
      lang_seq += [lang_map[language] for i in ipa]
      cleaned_text += ipa

    if language == 'JA':
      ipa = japanese_to_ipa(text)
      lang_seq += [lang_map[language] for i in ipa]
      cleaned_text += ipa

    elif language == 'ZH':
      ipa = chinese_to_ipa(text)
      lang_seq += [lang_map[language] for i in ipa]
      cleaned_text += ipa

    elif language == 'EN':
      ipa = english_to_ipa(text)
      lang_seq += [lang_map[language] for i in ipa]
      cleaned_text += ipa

  ipa = original_text[last_end:]

  lang_seq += [lang_map["other"] for i in ipa]
  cleaned_text += ipa

  assert len(cleaned_text) == len(lang_seq)
  return cleaned_text, lang_seq


def remove_invalid_text(cleaned_text, lang_seq):
  new_cleaned_text = ''
  new_lang_seq = []

  for symbol, la in zip(cleaned_text, lang_seq):
    if symbol not in symbols:
      print(cleaned_text)
      print("skip:", symbol)
      continue

    if la == lang_map["other"]:
      print("skip:", symbol)
      continue

    new_cleaned_text += symbol
    new_lang_seq.append(la)

  return new_cleaned_text, new_lang_seq


if __name__ == '__main__':
  print(clean_text("%[EN]Miss Radcliffe's letter had told him [EN]"))
  print(clean_text("[EN]Miss Radcliffe's letter had told him [EN]你好 hello[ZH]你好啊[ZH]"))
  print(clean_text("[P]ke3 % xian4 zai4 % jia4 ge2 % zhi2 jiang4 dao4 % yi2 wan4 duo1 $[P]"))
  print(clean_text("[ZH]可现在价格是降到一万多[ZH]"))
