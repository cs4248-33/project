import random
import spacy
import benepar
from nltk import ParentedTree
from typing import List

def constituency_sub_augmentation(
        inputs: List[str], 
        ooc_words: List[str], 
        n_generate: int=20000,
    ) -> List[str]:
    K = 200

    benepar.download('benepar_en3')
    
    nlp = spacy.load('en_core_web_md')
    nlp_en = spacy.load('en_core_web_md')
    nlp_en.add_pipe('benepar', config={'model': 'benepar_en3'})

    augmented_sentences = []
    count = 0

    for word in ooc_words:
        # Key: Ordered list of POS tags that uniquely identify a constituent
        # Value: List of tuple(ParentedTrees, index of constituent) that contain the constituent
        swap_tracker = {}
        sent_count = 0
        ooc_word = nlp(word)[0] # POS tag: ooc_word.tag_, text: ooc_word.text

        # Randomly select max K sentences from training set to augment
        random.shuffle(inputs)
        for input_sentence in inputs:
            if sent_count > K:
                break

            has_match = False
            parse_tree = None

            # Benepar might encounter errors with odd sentences that contain odd
            # punctuations etc. In such cases, we just skip them and pick the next random
            # sentence.
            try:
                en_parsed = nlp_en(input_sentence)
                en_parsed = list(en_parsed.sents)[0]
                parse_tree = ParentedTree.fromstring('(' + en_parsed._.parse_string + ')')
            except:
                # print("[const_aug]: skipping benepar error", input_sentence)
                continue

            # We identify the smallest possible constituents (height >= 3) that have 2-5 words
            for s in parse_tree.subtrees(lambda t: 3 <= t.height() <= 4 and 4 <= len(t.leaves()) <= 8):
                # Check if the constituent contains a word with the same POS as our ooc_word.
                # Add the ParentedTree to swap_tracker if true.
                for i, (_, tag) in enumerate(s.pos()):
                    if ooc_word.tag_ != tag:
                        continue

                    has_match = True
                    key = tuple(t[1] for t in s.pos())

                    # We want to control the size of each group to be maximum 6
                    # else we will have a huge factorial number of augmented data.
                    # if key in swap_tracker and len(swap_tracker[key]) >= 6:
                    #     break

                    # Swap OOC word in! We do a deep copy just in case.
                    s_copy = s.copy(deep=True)
                    for ss in s_copy.subtrees(lambda t: t.height() == 2):
                        if ss.label() == tag:
                            ss[0] = ooc_word.text

                    if key in swap_tracker:
                        swap_tracker[key].append((parse_tree, s_copy))
                    else:
                        swap_tracker[key] = [(parse_tree, s_copy)]

                    break
                
                if has_match:
                    break

            if has_match:
                sent_count += 1
                # print(f"[const_aug]: {sent_count}/{K} chosen for ({ooc_word.tag_}, {ooc_word.text})")
        
        # Drop those constituents with only 1 ParentedTree
        swap_tracker = {key: value for key, value in swap_tracker.items() if len(value) > 1}

        # Mix and match!
        for key, value in swap_tracker.items():
            if len(value) == 1:
                continue
            
            for curr_id, (_, curr_sub_pt) in enumerate(value):
                for match_id, (match_pt, _) in enumerate(value):
                    if curr_id == match_id:
                        continue

                    joined_words = ' '.join(match_pt.leaves())
                    joined_replacement = ' '.join(curr_sub_pt.leaves())
                    
                    # Get all subtrees that has the corresponding constituent to be replaced
                    matching_constituents = list(match_pt.subtrees(lambda t: tuple(tag[1] for tag in t.pos()) == key))

                    if len(matching_constituents) == 0:
                        continue
                    
                    const_to_replace = random.choice(matching_constituents)
                    joined_words = joined_words.replace(' '.join(const_to_replace.leaves()), joined_replacement, 1)

                    augmented_sentences.append(joined_words)
                    count += 1
                    if count % 100 == 0: print("Generated:", count, "sentences")
    
    if len(augmented_sentences) <= n_generate:
        return augmented_sentences
    
    return random.sample(augmented_sentences, n_generate)