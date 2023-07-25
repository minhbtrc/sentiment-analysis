import regex as re
import string
import json

from pipeline.dataset_manager.constants import COMMON_MAPPING_PATH


class Cleaner:
    def __init__(self):
        self.common_mapping = json.load(open(COMMON_MAPPING_PATH, "r", encoding="utf8"))

        self.emoji_pattern = re.compile("["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001f926-\U0001f937"
                                        u'\U00010000-\U0010ffff'
                                        u"\u200d"
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\u3030"
                                        u"\ufe0f"
                                        "]+", flags=re.UNICODE)

    def text_process(self, text):
        text = text.lower()
        # Remove emojis
        text = re.sub(self.emoji_pattern, " ", text)
        # Remove duplicate characters. shopppp -> shop
        text = re.sub(r'([a-z]+?)\1+', r'\1', text)
        # cam on shop.tam biet ==> cam on shop . tam biet
        text = re.sub(r"(\w)\s*([" + string.punctuation + "])\s*(\w)", r"\1 \2 \3", text)
        text = re.sub(r"(\w)\s*([" + string.punctuation + "])", r"\1 \2", text)
        # 99abcd => 99 abc
        # text = re.sub(r"(\d)([^\d.])", r"\1 \2", text)
        # text = re.sub(r"([^\d.])(\d)", r"\1 \2", text)
        # Remove duplicate punctuations
        text = re.sub(f"([{string.punctuation}])([{string.punctuation}])+", r"\1", text)
        # Mapping usual wrong word to formal word
        text = " ".join([self.common_mapping.get(word, word) for word in text.split()])
        while text.endswith(tuple(string.punctuation + string.whitespace)):
            text = text[:-1]
        while text.startswith(tuple(string.punctuation + string.whitespace)):
            text = text[1:]
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text