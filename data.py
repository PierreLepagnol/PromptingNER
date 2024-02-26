import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset

data_root = "data"


def get_row(func):
    def infunc(frame, i=None):
        if i is not None:
            frame = frame.loc[i, :]
        return func(frame, i=None)

    return infunc


def read_ob2(file_path):
    with open(file_path) as file:
        lines = file.readlines()
    sentences = []
    entities = []
    types = []
    exact_types = []
    data = []
    sub_entities = []
    sub_types = {}
    sub_exact_types = []
    words = ""
    curr_entity = ""
    curr_type = None

    for i, line in enumerate(lines):
        if line.strip() == "" or line == "\n" or i == len(lines) - 1:
            # save entity if it exists
            if curr_type is not None:
                sub_entities.append(curr_entity.strip())
                sub_types[curr_entity.strip()] = curr_type
                curr_entity = ""
                curr_type = None
            if words != "":
                sentences.append(words)
                entities.append(sub_entities)
                types.append(sub_types)
                exact_types.append(sub_exact_types)
                data.append([words, sub_entities, sub_types, sub_exact_types])
            sub_entities = []
            sub_types = {}
            sub_exact_types = []
            words = ""
            curr_entity = ""
            curr_type = None
        else:
            word, tag = line.split("\t")
            if words == "":
                words = word
            else:
                words = words + " " + word
            sub_exact_types.append(tag.strip())
            if tag.split() == "O" or "-" not in tag:  # if there was an entity before this then add it in full
                if curr_type is not None:
                    sub_entities.append(curr_entity.strip())
                    sub_types[curr_entity.strip()] = curr_type
                curr_entity = ""
                curr_type = None
            elif "B-" in tag or "I-" in tag:
                if "B-" in tag:
                    if curr_type is not None:
                        sub_entities.append(curr_entity.strip())
                        sub_types[curr_entity.strip()] = curr_type
                    curr_entity = word
                    curr_type = tag.split("-")[1].strip()
                else:  # I- in tag
                    if curr_type is None:
                        print(f"Should not be happening bug here")
                    curr_entity = curr_entity + " " + word
            else:
                main_type, subtype = tag.split(
                    "-"
                )  # must assume that if curr_type is not None then its the same one because FewNERD doesn't contain B, I information
                if subtype.strip() == "government/governmentagency":
                    subtype = "government"
                if curr_type is None:
                    curr_entity = word
                    curr_type = main_type + "-" + subtype.strip()  # can change to make it subtype if we want
                else:
                    curr_entity = curr_entity + " " + word

    df = pd.DataFrame(columns=["text", "entities", "types", "exact_types"], data=data)
    return df


def write_ob2(df, dataset_folder=None, filename=None):
    assert dataset_folder is not None and filename is not None
    os.makedirs(data_root + "/" + dataset_folder, exist_ok=True)
    with open(data_root + "/" + dataset_folder + "/" + filename + ".txt", "w") as f:
        for i in df.index:
            row = df.loc[i]
            sentence = row["text"]
            tokens = sentence.split(" ")
            if "true_tokens" in df.columns:
                tokens = row["true_tokens"]
            types = row["exact_types"]
            for j, word in enumerate(tokens):
                f.write(f"{word}\t{types[j]}\n")
            f.write("\n")
    return


def load_conll2003(split="validation") -> pd.DataFrame:
    dset = load_dataset("conll2003")[split]
    columns = ["text", "entities", "types", "exact_types"]
    #'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    conll_tag_map = {
        0: "none",
        1: "per",
        2: "per",
        3: "org",
        4: "org",
        5: "loc",
        6: "loc",
        7: "misc",
        8: "misc",
    }
    conll_fulltagmap = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC",
    }
    data = []
    for j in range(len(dset)):
        text = " ".join(dset[j]["tokens"])
        types = dset[j]["ner_tags"]
        sentence = text.split(" ")
        assert len(sentence) == len(types)
        entities = []
        d = {}
        subentities = ""
        curr_type = None
        exacts = []
        for i, tag in enumerate(types):
            exacts.append(conll_fulltagmap[tag])
            if tag == 0:
                if curr_type is not None:
                    entities.append(subentities)
                    d[subentities] = curr_type
                    curr_type = None
                    subentities = ""
            else:
                if tag in [1, 3, 5, 7]:
                    if curr_type is not None:
                        entities.append(subentities)
                        d[subentities] = curr_type
                    curr_type = conll_tag_map[tag]
                    subentities = sentence[i]
                else:
                    assert curr_type is not None
                    subentities = subentities + " " + sentence[i]
        data.append([text, entities, d, exacts])
    df = pd.DataFrame(columns=columns, data=data)
    return df


def load_genia(genia_path="data/Genia/Genia4ERtask1.iob2"):
    return read_ob2(genia_path)


def scroll(dataset, start=0, exclude=None):
    cols = dataset.columns
    for i in range(start, len(dataset)):
        s = dataset.loc[i]
        print(f"Item: {i}")
        for col in cols:
            if exclude is not None:
                if col in exclude:
                    continue
            print(f"{col}")
            print(s[col])
            print(f"XXXXXXXXXXXXXXX")
        inp = input("Continue?")
        if inp != "":
            return


def miniproc(x):
    if "-" in x:
        return x.split("-")[1]
    else:
        return x


def sample_all_types(dset, min_k=5):
    total_types = []
    for i in dset.index:
        types = list(set([miniproc(x) for x in dset.loc[i, "exact_types"]]))
        total_types.extend(types)
    total_types = list(set(total_types))
    done = False
    k = min_k
    i = 0
    minidset = None
    while not done:
        selected_types = []
        minidset = dset.sample(k).reset_index(drop=True)
        for i in minidset.index:
            types = list(set([miniproc(x) for x in minidset.loc[i, "exact_types"]]))
            selected_types.extend(types)
        selected_types = list(set(selected_types))
        if len(selected_types) == len(total_types):
            done = True
            break
        i += 1
        if (i + 1) % 10 == 0:
            k += 1
    return minidset


def save(func, name):
    for split in ["train", "validation", "test"]:
        dset = func(split=split)
        filename = split
        if filename == "validation":
            filename = "dev"
        write_ob2(dset, dataset_folder=name, filename=filename)
        minidset = sample_all_types(dset, min_k=5)
        write_ob2(minidset, name, "5shot" + filename)


@dataclass
class Paths:
    version: str
    split: str

    def __post_init__(self):
        directory = Path(f"/home/lepagnol/Documents/These/NER/MEDIA/media_{self.version}") / self.split

        self.datasets_text = directory / "seq.in"
        self.dataset_slot_filling = directory / "seq.out"
        self.dataset_intent = directory / "label"


def aggregate_to_dict(listoftypes) -> dict:
    if len(listoftypes) == 0:
        return {}
    else:
        # data_list = [
        #     ("je", "B-command-tache"),
        #     ("voudrais", "I-command-tache"),
        #     ("euh", "I-command-tache"),
        #     ("réserver", "I-command-tache"),
        #     ("pour", "B-localisation-ville"),
        #     ("la", "I-localisation-ville"),
        #     ("ville", "I-localisation-ville"),
        #     ("de", "I-localisation-ville"),
        #     ("Nice", "I-localisation-ville"),
        #     ("du", "B-temps-date"),
        #     ("premier", "I-temps-date"),
        #     ("au", "B-temps-date"),
        #     ("trois", "I-temps-date"),
        #     ("novembre", "I-temps-date"),
        # ]

        return aggregate_phrases_corrected(listoftypes)


def aggregate_phrases_corrected(list_of_tuples):
    aggregated_dict = {}
    current_phrase = ""
    current_tag = ""
    for word, tag in list_of_tuples:
        if tag.startswith("B-"):  # Beginning of a new phrase
            if current_phrase:  # Save the previous phrase if it exists
                aggregated_dict[current_phrase] = current_tag  # Assign tag to phrase
            current_phrase = word  # Start a new phrase
            current_tag = tag[2:]  # Update the current tag without the B-/I- prefix
        else:  # Intermediate word of the current phrase
            current_phrase += " " + word
    # Add the last phrase to the dictionary
    if current_phrase:
        aggregated_dict[current_phrase] = current_tag
    return aggregated_dict


def load_media(split="test", version="original"):
    assert version in ["original", "speechbrain_full", "speechbrain_relax"]

    paths = Paths(version=version, split=split)

    with open(paths.datasets_text, "r") as file:
        texts = file.readlines()

    with open(paths.dataset_slot_filling, "r") as file:
        dataset_slot_filling = file.readlines()

    with open(paths.dataset_intent, "r") as file:
        dataset_intent = file.readlines()

    print("len(texts):", len(texts))
    print("len(dataset_slot_filling):", len(dataset_slot_filling))
    print("len(dataset_intent):", len(dataset_intent))

    data = []
    counter = 0
    for text, slot_filling, intent in zip(texts, dataset_slot_filling, dataset_intent):
        # print(counter)
        if counter == 3:
            print(text, slot_filling, intent)
        text_list = text.replace("\n", "").split()
        slot_filling = slot_filling.replace("\n", "").split()
        intent = intent.replace("\n", "").split()
        true_types = []
        counter += 1

        for i, word in enumerate(text_list):
            if slot_filling[i] != "O":
                true_types.append((word, slot_filling[i]))
        true_types = aggregate_to_dict(true_types)

        data.append([text, true_types, slot_filling, intent])

    data[3]
    df = pd.DataFrame(columns=["text", "entities", "types", "intent"], data=data)  # "exact_types"
    return df


if __name__ == "__main__":
    # load_conll2003("test")
    load_media(split="test")
    # save(load_fabner, "fabner")
    # save(load_tweetner, "tweetner")
