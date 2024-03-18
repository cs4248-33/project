import re

zh_path = '../data/train.tags.en-zh.zh'
en_path = '../data/train.tags.en-zh.en'

zh_output_path = '../data/train.zh'
en_output_path = '../data/train.en'

title_pattern = r'<title>(.*?)</title>'
description_pattern = r'<description>TED Talk Subtitles and Transcript: (.*?)</description>'
tags_to_skip = ['doc', 'url', 'keywords', 'speaker', 'talkid', 'reviewer', 'translator']

def remove_title_tags(text):
    cleaned_text = re.sub(title_pattern, '', text)
    return cleaned_text

with open(zh_path, 'r') as zh_file, open(en_path, 'r') as en_file, open(zh_output_path, 'w') as zh_out, open(en_output_path, 'w') as en_out:
    for line1, line2 in zip(zh_file, en_file):
        zh_line = line1.strip()
        en_line = line2.strip()

        # Check if it is a title tag
        zh_is_title = re.search(title_pattern, zh_line)
        en_is_title = re.search(title_pattern, en_line)

        # Check if it is a description tag
        zh_is_desc = re.search(description_pattern, zh_line)
        en_is_desc = re.search(description_pattern, en_line)

        if zh_is_title and en_is_title:
            zh_line = zh_is_title.group(1)
            en_line = en_is_title.group(1)
            pass
        elif zh_is_desc and en_is_desc:
            zh_line = zh_is_desc.group(1)
            en_line = en_is_desc.group(1)
        
        if any(tag in zh_line for tag in tags_to_skip):
            continue
        
        zh_out.write(f'{zh_line}\n')
        en_out.write(f'{en_line}\n')
