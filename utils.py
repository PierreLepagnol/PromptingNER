
import string
import re
from numpy.random import choice
from nltk.corpus import stopwords


def find_nth_str(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def find_nth_list(haystack, needle, n):
    start = haystack.index(needle)
    while start >= 0 and n > 1:
        start = haystack.index(needle, start+1)
        n -= 1
    return start

def find_nth_list_subset(haystack, needle, n):
    if n < 0:
        return -1
    if n == 0:
        n = 1
    found = []
    needle_size = len(needle.split(" "))
    for i in range(len(haystack)):
        sliced = " ".join(haystack[i:i+needle_size])
        if needle == sliced:
            found.append(i)
    if len(found) > n:
        return -1
    else:
        return found[n-1]


def separate_single_multi(l):
    singles, multis = [], []
    for item in l:
        i = item.strip()
        if " " in i:
            multis.append(i)
        else:
            singles.append(i)
    return singles, multis


def verbose(func):
    def inner(*args, **kwargs):
        if kwargs.get("verbose", False):
            i = kwargs.get("indent_level", 0)
            indent = "\t"*i
            print(f"{indent}{args[0].strip()}")
        return func(*args, **kwargs)
    return inner


class AnswerMapping:
    @staticmethod
    @verbose
    def get_numbered_list_items(output, verbose=False, indent_level=0):
        final = []
        if "\n" in output:
            candidates = output.split("\n")
            for cand in candidates:
                c = cand.strip()
                if c.lower().strip() in ["", "answer:"]:
                    pass
                elif re.match(r"\d+[.)]+ *", c):
                    start = 0
                    while c[start].isnumeric() or c[start] == '.':
                        start += 1
                    final.append(c[start:].strip())
                else:
                    print(f"Unable to match nonempty {c}")
                    pass
        else:
            candidates = re.split(r"\d+[.)]", output)
            for cand in candidates:
                c = cand.strip()
                if c.lower().strip() in ["", "answer:"]:
                    pass
                else:
                    final.append(c)
        return final

    @staticmethod
    @verbose
    def get_true_or_false(output, default=True, verbose=False, indent_level=0):
        output = output.lower()
        true_condition = "yes " in output or "yes." in output or "true" in output
        false_condition = "no " in output or "no." in output or "false" in output

        if true_condition and not false_condition:
            return True
        elif false_condition and not true_condition:
            return False
        else:
            if not true_condition and not false_condition:
                print(f"Unable to map {output} to True or False")
            else:
                print(f"Mapping {output} to both True or False")
            return default

    @staticmethod
    @verbose
    def exemplar_format_list(output, verbose=False, indent_level=0, separator='|', true_only=True, identify_types=False):
        if "\n" in output:
            listed = AnswerMapping.get_numbered_list_items(output, verbose=False, indent_level=indent_level+1)
        else:
            listed = []
            if "1" in output:
                split = re.split(r"\d+[.)]", output)
                for item in split:
                    if item.strip().lower() == "" or "answer" in item.strip().lower():
                        pass
                    else:
                        listed.append(item.strip())
        final = []
        typestring = []
        for option in listed:
            if separator in option:
                split = option.split(separator)
                explanation = None
                if len(split) == 1:
                    print(f"Got only one value for {option} with separator '{separator}'")
                    continue
                elif len(split) == 2:
                    entity, depends = split
                    if depends.strip().lower() in ["true", "false"]:
                        status = depends
                    else:
                        status = "true"
                        explanation = depends
                elif len(split) == 3:
                    entity, status, explanation = split
                else:
                    entity, status = split[0], split[1]
                    print(f"Got more than 3 values for {option} with separator '{separator}'")
                if status.strip().lower() == "true" or not true_only:
                    if explanation is not None:
                        typestring.append(explanation.strip())
                    final.append(entity.strip().lower())
                else:
                    pass
            else:
                final.append(option.strip().lower())
        if not identify_types:
            return final
        else:
            return final, typestring


class Parameters:
    devices = ["cuda:0"]

    @staticmethod
    def get_device_ints(limit=3):
        assert "cpu" not in Parameters.devices[:limit]
        final = []
        for item in Parameters.devices[:limit]:
            if isinstance(item, int):
                final.append(item)
            elif item.isnumeric():
                final.append(int(item))
            else:
                f, l = item.split(":")
                final.append(int(l))
        return final
