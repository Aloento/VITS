import argparse
from os import path

import tqdm

import text.cleaners
from utils import load_filepaths_and_text

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="clr.csv")
  parser.add_argument("--text_index", default=2, type=int)
  parser.add_argument("--filelists", nargs="+", default=[
    "filelists/train.csv",
    "filelists/val.csv"
  ])
  parser.add_argument("--text_cleaners", nargs="+", default=["cje_cleaner"])

  args = parser.parse_args()

  for filelist in args.filelists:
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)

    for i in tqdm.tqdm(range(len(filepaths_and_text))):
      original_text = filepaths_and_text[i][args.text_index]
      cleaned_text, lang_seq = text.cleaners.clean_text(original_text)
      filepaths_and_text[i][args.text_index] = cleaned_text
      filepaths_and_text[i].append(" ".join([str(lang) for lang in lang_seq]))

    new_filelist = path.splitext(filelist)[0] + "." + args.out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
      f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
