import string

import numpy as np
from nltk.corpus import stopwords
from transformers import PreTrainedModel, PreTrainedTokenizer

import myutils

# from models import OpenAIGPT
from myutils import AnswerMapping


class BaseAlgorithm:
    defn = (
        "An entity is an object, place, individual, being, title, proper noun or process that has a distinct and "
        "independent existence. The name of a collection of entities is also an entity. Adjectives, verbs, numbers, "
        "adverbs, abstract concepts are not entities. Dates, years and times are not entities"
    )

    chatbot_init = "You are an entity recognition system. "
    entity_token_task = "In the sentence '[sent]'. The phrase '[token]' is an entity of type [type]. In one line explain why. \nAnswer: The phrase '[token]' is an entity of type [type] because"
    nonentity_token_task = "In the sentence '[sent]'. The phrase '[token]' is not an entity. In one line explain why. \nAnswer: The phrase '[token]' is not an entity because"

    # if [] = n then there are O(n^2) phrase groupings

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        split_phrases=False,
        identify_types=True,
        resolve_disputes=True,
    ):
        self.defn = self.defn
        self.para = None

        self.model = model
        self.tokenizer = tokenizer

        self.split_phrases = split_phrases
        self.exemplar_task = None
        self.format_task = None
        self.whole_task = None
        self.identify_types = identify_types
        self.resolve_disputes = resolve_disputes

    def set_para(self, para):
        self.para = para

    def set_model_fn(self, model_fn):
        self.model_fn = model_fn

    @staticmethod
    def clean_output(answers, typestrings=None):
        if typestrings is None:
            answers = list(set(answers))
            for trivial in ["", " ", ".", "-"] + stopwords.words("english"):
                while trivial in answers:
                    answers.remove(trivial)
        else:
            new_answers = []
            new_typestrings = []
            for i, ans in enumerate(answers):
                if ans in new_answers:
                    continue
                if ans in ["", " ", ".", "-"] + stopwords.words("english"):
                    continue
                new_answers.append(ans)
                new_typestrings.append(typestrings[i])
        for i in range(len(answers)):
            ans = answers[i]
            if "(" in ans:
                ans = ans[: ans.find("(")]
            ans = ans.strip().strip("".join(string.punctuation)).strip()
            answers[i] = ans
        if typestrings is None:
            return answers
        else:
            return answers, typestrings


class Algorithm(BaseAlgorithm):
    def query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        generated_ids = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]

    def perform_span(self, true_tokens=None, verbose=False):
        assert self.identify_types and not self.split_phrases
        answers, typestrings, metadata = self.perform(verbose=verbose, deduplicate=False)
        return self.parse_span(answers, typestrings, metadata, true_tokens=true_tokens)

    def parse_span(self, answers, typestrings, metadata, true_tokens=None):
        para = self.para.lower()

        if true_tokens is not None:
            para_words = [token.lower() for token in true_tokens]
        else:
            para_words = para.split(" ")

        span_pred = ["O" for word in para_words]
        completed_answers = []
        split_tokens = ["'s", ":"]

        for i, answer in enumerate(answers):
            answer = answer.strip().lower()  # take any whitespace out and lowercase for matching

            if "(" in answer:
                answer = answer[: answer.find("(")].strip()  # in case some type annotation is stuck here

            types = typestrings[i]

            if "(" in types and ")" in types:
                types = types[types.find("(") + 1 : types.find(")")]
            else:
                continue

            answer_token_split = answer

            for token in split_tokens:
                answer_token_split = (" " + token).join(answer_token_split.split(token))
            exists = answer in para or answer_token_split in para
            answer_multi_word = len(answer.split(" ")) > 1

            if not exists:
                continue

            if not answer_multi_word:
                if answer not in para_words:
                    continue
                multiple = para.count(answer) > 1
                if not multiple:  # easiest case word should be in para_words only once
                    index = para_words.index(answer)
                else:  # must find which occurance this is
                    n_th = completed_answers.count(answer.strip()) + 1
                    index = myutils.find_nth_list(para_words, answer, n_th)
                if span_pred[index] == "O":
                    if "-" in types:  # then its FEWNERD
                        span_pred[index] = types
                    else:
                        span_pred[index] = "B-" + types
                completed_answers.append(answer)
            else:
                for token in split_tokens:
                    if token in answer:
                        answer = (" " + token).join(answer.split(token))
                answer_words = answer.split(" ")
                multiple = para.count(answer) > 1
                n_th = completed_answers.count(answer.strip()) + 1
                index = myutils.find_nth_list_subset(para_words, answer, n_th)
                end_index = index + len(answer_words)
                if "-" in types:  # then its FEWNERD
                    span_pred[index] = types
                else:
                    span_pred[index] = "B-" + types
                for j in range(index + 1, end_index):
                    if "-" in types:  # then its FEWNERD
                        span_pred[j] = types
                    else:
                        span_pred[j] = "I-" + types
                completed_answers.append(answer)

        return span_pred, metadata

    def perform_openai(self, verbose=True, deduplicate=True):
        if not self.identify_types:
            if self.model_fn.is_chat():
                answers, metadata = self.perform_chat_query(verbose=verbose)
            else:
                answers, metadata = self.perform_single_query(verbose=verbose)
        else:
            if self.model_fn.is_chat():
                answers, typestrings, metadata = self.perform_chat_query(verbose=verbose)
            else:
                answers, typestrings, metadata = self.perform_single_query(verbose=verbose)

        return answers, typestrings, metadata

    def perform(self, verbose=True, deduplicate=True):
        """

        :param model:
        :param paragraph:
        :return:
        """
        # if isinstance(self.model, OpenAIGPT):
        #     self.perform_openai(verbose=verbose, deduplicate=deduplicate)
        # else:
        if self.identify_types:
            answers, typestrings, metadata = self.perform_single_query(verbose=verbose)
        else:
            answers, metadata = self.perform_single_query(verbose=verbose)

        if not self.identify_types:
            answers = list(set(answers))

        if self.split_phrases:
            new_answers = []
            if self.identify_types:
                new_typestrings = []
            for i, answer in enumerate(answers):
                if " " not in answer:
                    new_answers.append(answer)
                    if self.identify_types:
                        new_typestrings.append(typestrings[i])
                else:
                    minis = answer.split(" ")
                    for mini in minis:
                        new_answers.append(mini)
                        if self.identify_types:
                            new_typestrings.append(typestrings[i])
            answers = new_answers
            if self.identify_types:
                typestrings = new_typestrings

        if deduplicate:
            if self.identify_types:
                answers, typestrings = BaseAlgorithm.clean_output(answers, typestrings)
            else:
                answers = BaseAlgorithm.clean_output(answers)

        if not self.identify_types:
            return answers, metadata
        else:
            return answers, typestrings, metadata

    def perform_single_query(self, verbose=True):
        # Few-shot
        if self.exemplar_task is not None:
            full_prompt = self.defn + "\n" + self.exemplar_task + f" '{self.para}' \nAnswer:"
            output = self.query(full_prompt)
            final = AnswerMapping.exemplar_format_list(
                output, identify_types=self.identify_types, verbose=verbose
            )
        # Zero-shot
        else:
            full_prompt = self.defn + "\n" + self.format_task + f"\nParagraph: {self.para} \nAnswer:"
            output = self.query(full_prompt)
            final = AnswerMapping.exemplar_format_list(
                output, identify_types=self.identify_types, verbose=verbose
            )

        if self.identify_types:
            final, typestrings = final
            return final, typestrings, output
        else:
            return final, output

    def perform_chat_query(self, verbose=True):
        if self.exemplar_task is not None:
            system_msg = self.chatbot_init + self.defn + " " + self.whole_task
            msgs = [(system_msg, "system")]
            for exemplar in self.exemplars:
                if "Answer:" not in exemplar:
                    raise ValueError(
                        f"Something is wrong, exemplar: \n{exemplar} \n Does not have an 'Answer:'"
                    )
                ans_index = exemplar.index("Answer:")
                msgs.append((exemplar[: ans_index + 7].strip(), "user"))
                msgs.append((exemplar[ans_index + 7 :].strip(), "assistant"))
            msgs.append((f"\nParagraph: {self.para} \nAnswer:", "user"))
            output = self.model_fn(msgs)
            final = AnswerMapping.exemplar_format_list(
                output, identify_types=self.identify_types, verbose=verbose
            )
        else:
            system_msg = self.chatbot_init + self.defn + " " + self.format_task
            msgs = [(system_msg, "system"), (f"\nParagraph: {self.para} \nAnswer:", "user")]
            output = self.model_fn(msgs)
            final = AnswerMapping.exemplar_format_list(
                output, identify_types=self.identify_types, verbose=verbose
            )
        if self.identify_types:
            final, typestrings = final
        if not self.identify_types:
            return final, output
        else:
            return final, typestrings, output

    def get_annotation(self, token, ner_label):
        if ner_label == "O":
            task_string = self.nonentity_token_task.replace("[sent]", self.para)
            task_string = task_string.replace("[token]", token)
        else:
            task_string = self.entity_token_task.replace("[sent]", self.para)
            task_string = task_string.replace("[token]", token)
            task_string = task_string.replace("[type]", ner_label)

        if self.model_fn.is_chat():
            msgs = [(self.defn, "system"), (task_string, "user")]
            output = self.model_fn(msgs)
        else:
            task_string = self.defn + "\n" + task_string
            output = self.model_fn(task_string)

        return output

    def generate_annotations(self, tokens, ner_labels, max_falses=3):
        false_indices = []
        annots = []
        for i, token in enumerate(tokens):
            if ner_labels[i] != "O":
                annot = self.get_annotation(token, ner_labels[i])
                annots.append(annot)
            else:
                if (
                    token.strip().strip(string.punctuation).strip() == ""
                    or token.strip() in stopwords.words("english")
                    or token.isnumeric()
                ):
                    annots.append(None)
                else:
                    false_indices.append(i)
                    annot = self.get_annotation(token, "O")
                    annots.append(annot)

        if len(false_indices) > max_falses:
            false_indices = np.random.choice(false_indices, max_falses, replace=False)
            false_indices.sort()

        annot_str = "Answer: \n"
        no = 1

        for i, token in enumerate(tokens):
            if annots[i] is None:
                pass
            else:
                appendage = "\n" + f"{no}. {token} | {ner_labels[i] != 'O'} | {annots[i]}"
                if ner_labels[i] != "O":
                    if ner_labels[i][:2] in ["B-", "I-"]:
                        label = ner_labels[i][2:]
                    else:
                        label = ner_labels[i]
                    annot_str = annot_str + appendage + f"({label})"
                    no += 1
                else:
                    if i in false_indices:
                        annot_str = annot_str + appendage
                        no += 1

        return annot_str


class MultiAlgorithm(Algorithm):
    def perform_span(self, true_tokens=None, resolve_disputes=False, verbose=False):
        assert self.identify_types and not self.split_phrases
        answers, typestrings, metadata = self.perform(verbose=verbose, deduplicate=False)
        span_pred, metadata = self.parse_span(
            answers, typestrings, metadata, query=True, true_tokens=true_tokens, verbose=verbose
        )
        return span_pred, metadata

    def parse_span(self, answers, typestrings, metadata, true_tokens=None, query=False, verbose=False):
        para = self.para.lower()
        if true_tokens is not None:
            para_words = [token.lower() for token in true_tokens]
        else:
            para_words = para.split(" ")
        span_pred = ["O" for word in para_words]
        completed_answers = []
        split_tokens = ["'s", ":"]
        for i, answer in enumerate(answers):
            answer = answer.strip().lower()  # take any whitespace out and lowercase for matching
            if "(" in answer:
                answer = answer[: answer.find("(")].strip()  # in case some type annotation is stuck here
            if not self.resolve_disputes and query:
                types = self.get_type(answer, verbose=verbose)
                if types == -1:
                    types = typestrings[i]
                    if "(" in types and ")" in types:
                        types = types[types.find("(") + 1 : types.find(")")]
                    else:
                        continue
            else:
                types = typestrings[i]
                if "(" in types and ")" in types:
                    types = types[types.find("(") + 1 : types.find(")")]
                else:
                    continue
                if self.resolve_disputes:
                    other_types = self.get_type(answer, verbose=verbose)
                    if types != other_types:
                        types = self.resolve_dispute(answer, types, other_types, verbose=verbose)
                        if types == -1:
                            types = typestrings[i]
                            if "(" in types and ")" in types:
                                types = types[types.find("(") + 1 : types.find(")")]
                            else:
                                continue

            answer_token_split = answer
            for token in split_tokens:
                answer_token_split = (" " + token).join(answer_token_split.split(token))
            exists = answer in para or answer_token_split in para
            answer_multi_word = len(answer.split(" ")) > 1
            if not exists:
                continue
            if not answer_multi_word:
                if answer not in para_words:
                    continue
                multiple = para.count(answer) > 1
                if not multiple:  # easiest case word should be in para_words only once
                    index = para_words.index(answer)
                else:  # must find which occurance this is
                    n_th = completed_answers.count(answer.strip()) + 1
                    index = myutils.find_nth_list(para_words, answer, n_th)
                if span_pred[index] == "O":
                    if "-" in types:  # then its FEWNERD
                        span_pred[index] = types
                    else:
                        span_pred[index] = "B-" + types
                completed_answers.append(answer)
            else:
                for token in split_tokens:
                    if token in answer:
                        answer = (" " + token).join(answer.split(token))
                answer_words = answer.split(" ")
                multiple = para.count(answer) > 1
                n_th = completed_answers.count(answer.strip()) + 1
                index = myutils.find_nth_list_subset(para_words, answer, n_th)
                end_index = index + len(answer_words)
                if "-" in types:  # then its FEWNERD
                    span_pred[index] = types
                else:
                    span_pred[index] = "B-" + types
                for j in range(index + 1, end_index):
                    if "-" in types:  # then its FEWNERD
                        span_pred[j] = types
                    else:
                        span_pred[j] = "I-" + types
                completed_answers.append(answer)
        return span_pred, metadata

    def get_type(self, phrase, verbose=False):
        task = self.type_task
        afterphrase = f"Entity Phrase: {phrase}"
        if self.model_fn.is_chat():
            exemplars = self.type_exemplars
            answer = self.template_chat_query(task, exemplars, afterphrase, verbose=verbose)
        else:
            task = self.type_task_exemplars
            answer = self.template_single_query(task, afterphrase, verbose=verbose)
        if "(" in answer and ")" in answer:
            start = answer.find("(")
            end = answer.find(")")
            return answer[start + 1 : end]
        else:
            return -1

    def resolve_dispute(self, phrase, option1, option2, verbose=False):
        task = self.dispute_task
        afterphrase = f"Entity Phrase: {phrase}, Options: ({option1}), ({option2})"
        if self.model_fn.is_chat():
            exemplars = self.dispute_exemplars
            answer = self.template_chat_query(task, exemplars, afterphrase, verbose=verbose)
        else:
            task = self.dispute_task_exemplars
            answers = self.template_single_query(task, afterphrase, verbose=verbose)
        if "(" in answer and ")" in answer:
            start = answer.find("(")
            end = answer.find(")")
            return answer[start + 1 : end]
        else:
            return -1

    def template_chat_query(self, task, exemplars, afterphrase, verbose=False):
        system_msg = self.chatbot_init + self.defn + " " + task
        msgs = [(system_msg, "system")]
        for exemplar in exemplars:
            if "Answer:" not in exemplar:
                raise ValueError(f"Something is wrong, exemplar: \n{exemplar} \n Does not have an 'Answer:'")
            ans_index = exemplar.index("Answer:")
            msgs.append((exemplar[: ans_index + 7].strip(), "user"))
            msgs.append((exemplar[ans_index + 7 :].strip(), "assistant"))
        msgs.append((f"\nParagraph: {self.para} \n{afterphrase} \nAnswer:", "user"))
        output = self.model_fn(msgs)
        if verbose:
            print(output)
        return output

    def template_single_query(self, task, afterphrase, verbose=False):
        task = self.defn + "\n" + task + f" '{self.para}' \n{afterphrase} \nAnswer:"
        output = self.model_fn(task)
        if verbose:
            print(output)
        return output


class Config:
    """
    This class is used to configure the task format for the algorithm.
    It contains formats for zero-shot and few-shot learning, with and without chain of thoughts,
    and with and without true/false indicators.
    """

    concept_definitions = ""

    cot_format = """
    Format: 
    
    1. First Candidate | True | Explanation why the word is an entity (entity_type)
    2. Second Candidate | False | Explanation why the word is not an entity (entity_type)
    """

    no_tf_format = """
    1. First Entity | Explanation why the word is an entity (entity_type)
    2. Second Entity | Explanation why the word is not an entity (entity_type)
    """

    tf_format = """
    Format: 

    1. First Candidate | True | (entity_type)
    2. Second Candidate | False | (entity_type)
    """

    exemplar_format = """
    Format:    
    
    1. First Entity | (entity_type)
    2. Second Entity | (entity_type)
    """

    def zs_format_without_chain_of_thoughts(self, alg: Algorithm, tf=True):
        """
        Set the task format for zero-shot learning without chain of thoughts.

        Args:
            alg (Algorithm): The algorithm object to set the task format for.
            tf (bool, optional): Whether to include true/false indicators in the format. Defaults to True.
        """
        whole_task = "Q: Given the paragraph below, identify the list of entities " "Answer in the format: \n"
        if not tf:
            alg.format_task = whole_task + self.exemplar_format
        else:
            alg.format_task = whole_task + self.tf_format

    def zs_format_with_chain_of_thoughts(self, alg: Algorithm, tf=True):
        """
        Set the task format for zero-shot learning with chain of thoughts.

        Args:
            alg (Algorithm): The algorithm object to set the task format for.
            tf (bool, optional): Whether to include true/false indicators in the format. Defaults to True.
        """
        if tf:
            whole_task = (
                "Q: Given the paragraph below, identify a list of possible entities "
                "and for each entry explain why it either is or is not an entity. Answer in the format: \n"
            )

            alg.format_task = whole_task + self.cot_format
        else:
            whole_task = (
                "Q: Given the paragraph below, identify a list of entities "
                "and for each entry explain why it is an entity. Answer in the format: \n"
            )

            alg.format_task = whole_task + self.no_tf_format

    def fs_format_without_chain_of_thoughts(self, alg: Algorithm, tf=True):
        """
        Set the task format for few-shot learning without chain of thoughts.

        Args:
            alg (Algorithm): The algorithm object to set the task format for.
            tf (bool, optional): Whether to include true/false indicators in the format. Defaults to True.
        """
        whole_task = "Q: Given the paragraph below, identify the list of entities \nParagraph:"
        exemplar_construction = ""
        if not tf:
            e_list = self.exemplars
        else:
            e_list = self.tf_exemplars
        alg.whole_task = whole_task
        alg.exemplars = e_list
        for exemplar in e_list:
            exemplar_construction = exemplar_construction + whole_task + "\n"
            exemplar_construction = exemplar_construction + exemplar + "\n"
        exemplar_construction = exemplar_construction + whole_task + "\n"
        alg.exemplar_task = exemplar_construction

    def fs_format_with_chain_of_thoughts(self, alg: Algorithm, tf=True):
        """
        Set the task format for few-shot learning with chain of thoughts.

        Args:
            alg (Algorithm): The algorithm object to set the task format for.
            tf (bool, optional): Whether to include true/false indicators in the format. Defaults to True.
        """
        # Use TRUE | FALSE
        if tf:
            whole_task = (
                "Q: Given the paragraph below, identify a list of possible entities "
                "and for each entry explain why it either is or is not an entity. \nParagraph:"
            )
            alg.whole_task = whole_task
            alg.exemplars = self.cot_exemplars
            exemplar_construction = ""
            for exemplar in self.cot_exemplars:
                exemplar_construction = exemplar_construction + whole_task + "\n"
                exemplar_construction = exemplar_construction + exemplar + "\n"
            exemplar_construction = exemplar_construction + whole_task + "\n"
            alg.exemplar_task = exemplar_construction
        else:
            whole_task = (
                "Q: Given the paragraph below, identify a list of entities "
                "and for each entry explain why it is an entity. \nParagraph:"
            )
            alg.whole_task = whole_task
            alg.exemplars = self.no_tf_exemplars
            exemplar_construction = ""
            for exemplar in self.no_tf_exemplars:
                exemplar_construction = exemplar_construction + whole_task + "\n"
                exemplar_construction = exemplar_construction + exemplar + "\n"
            exemplar_construction = exemplar_construction + whole_task + "\n"
            alg.exemplar_task = exemplar_construction

    def set_config(self, alg: Algorithm, exemplar=True, coT=True, tf=True, defn=True):
        """
        Set the configuration for the algorithm.

        Args:
            alg (Algorithm): The algorithm object to set the configuration for.
            exemplar (bool, optional): Whether to use few-shot learning. Defaults to True.
            coT (bool, optional): Whether to include chain of thoughts in the format. Defaults to True.
            tf (bool, optional): Whether to include true/false indicators in the format. Defaults to True.
            defn (bool, optional): Whether to include definitions in the task. Defaults to True.
        """
        if isinstance(alg, MultiAlgorithm):
            coT = False
            tf = False
            exemplar = True
            type_task = "Q: Given the paragraph below and the entity phrase, identify what type the entity is \nParagraph:"
            alg.type_exemplars = self.type_exemplars
            exemplar_construction = ""
            for exemplar in self.type_exemplars:
                exemplar_construction = exemplar_construction + type_task + "\n"
                exemplar_construction = exemplar_construction + exemplar + "\n"
            exemplar_construction = exemplar_construction + type_task + "\n"
            alg.type_task_exemplars = exemplar_construction
            alg.type_task = type_task

            dispute_task = "Q: Given the paragraph below, the entity phrase and two proposed entity types, identify what the actual type of the entity is \nParagraph:"
            alg.dispute_exemplars = self.dispute_exemplars
            exemplar_construction = ""
            for exemplar in self.dispute_exemplars:
                exemplar_construction = exemplar_construction + dispute_task + "\n"
                exemplar_construction = exemplar_construction + exemplar + "\n"
            exemplar_construction = exemplar_construction + dispute_task + "\n"
            alg.dispute_task_exemplars = exemplar_construction
            alg.dispute_task = dispute_task

        # Whether to include the definition in the task
        if defn:
            alg.defn = self.defn
        else:
            alg.defn = ""

        # Fewshot or Zero-shot ?
        if not exemplar:
            # Zero-shot
            alg.exemplar_task = None
            if coT:
                self.zs_format_with_chain_of_thoughts(alg, tf)
            else:
                self.zs_format_without_chain_of_thoughts(alg, tf)
        else:
            # Few-shot
            alg.format_task = None
            # Use Chain-of-Thoughts
            if coT:
                self.fs_format_with_chain_of_thoughts(alg, tf)
            else:
                self.fs_format_without_chain_of_thoughts(alg, tf)

    def autogenerate_annotations(self, alg: Algorithm, texts, tokens, labels, max_examples=3):
        """
        Generate annotations for the given texts and tokens.

        Args:
            alg (Algorithm): The algorithm object to generate annotations for.
            texts (list): The list of texts to generate annotations for.
            tokens (list): The list of tokens to generate annotations for.
            labels (list): The list of labels to generate annotations for.
            max_examples (int, optional): The maximum number of examples to generate annotations for. Defaults to 3.
        """
        cot_exemplars = []
        for i in range(len(texts[:max_examples])):
            text = texts[i]
            token = tokens[i]
            label = labels[i]
            alg.set_para(text)
            exemplar = text + "\n" + alg.generate_annotations(token, label)
            cot_exemplars.append(exemplar)
        self.cot_exemplars = cot_exemplars


class AtisConfig(Config):
    def __init__(self):
        raise NotImplementedError("AtisConfig is not yet implemented")


class MediaConfig(Config):
    defn = """An entity is a person (PER), title, named organization (ORG), location (LOC), country (LOC) or nationality (MISC).
        Names, first names, last names, countries are entities.
        Nationalities are entities even if they adjectives.
        Sports, sporting events, adjectives, verbs, adverbs, abstract concepts, sports, are not entities.
        Dates, years and times are not Possessive words like I, you, him and me are not If a sporting team has the name of their location and the location is used to refer to the  it is an entity which is an organisation, not a location"""

    defn = "An entity is a person (PER), title, named organization (ORG), location (LOC), country (LOC) or nationality (MISC)."

    cot_exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
    
    Answer:
    1. bowling | False | as it is an action
    2. Somerset | True | Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)
    3. 83 | False | as it is a number 
    4. morning | False| as it represents a time of day, with no distinct and independant existence
    5. Grace Road | True | the game is played at Grace Road, hence it is a place or location (LOC)
    6. Leicestershire | True | is the name of a cricket team that is based in the town of Leicestershire, hence it is an organisation (ORG). 
    7. first innings | False | as it is an abstract concept of a phase in play of cricket
    8. England | True | as it is a place or location (LOC)
    9. Andy Caddick | True | as it is the name of a person. (PER) 
    """
    cot_exemplar_2 = """
    Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
    
    Answer:
    1. Their | False | as it is a possessive pronoun
    2. stay | False | as it is an action
    3. title rivals | False | as it is an abstract concept
    4. Essex | True |  Essex are title rivals is it a sporting team organisation not a location (ORG)
    5. Derbyshire | True |  Derbyshire are title rivals is it a sporting team organisation not a location (ORG)
    6. Surrey | True |  Surrey are title rivals is it a sporting team organisation not a location (ORG)
    7. victory | False | as it is an abstract concept
    8. Kent | True |  Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
    9. Nottinghamshire | True |  Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
    
    """

    cot_exemplar_3 = """
    But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .
    
    Answer:
    1. money | False | as it is not a named person, organization or location
    2. savings account | False | as it is not a person, organization or location
    3. 5.3 | False | as it is a number
    4. June | False | as it is a date
    5. July | False | as it is a date
    """

    cot_exemplars = [cot_exemplar_1, cot_exemplar_2, cot_exemplar_3]

    no_tf_exemplar_1 = """
 After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
 
        Answer:
        1. Somerset | Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)
        2. Grace Road | the game is played at Grace Road, hence it is a place or location (LOC)
        3. Leicestershire | is the name of a cricket team that is based in the town of Leicestershire, hence it is an organisation (ORG). 
        4. England | as it is a place or location (LOC)
        5. Andy Caddick | as it is the name of a person. (PER) 
        """
    no_tf_exemplar_2 = """
        Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
        
        Answer:
        1. Essex | since Essex are title rivals is it a sporting team organisation not a location (ORG)
        2. Derbyshire | since Derbyshire are title rivals is it a sporting team organisation not a location (ORG)
        3. Surrey | since Surrey are title rivals is it a sporting team organisation not a location (ORG)
        4. Kent | since Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
        5. Nottinghamshire | since Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
        """

    no_tf_exemplar_3 = """
        But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .

        Answer:
        1. 

        """
    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2, no_tf_exemplar_3]

    tf_exemplar_1 = """
        After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .

        Answer:
        1. bowling | False | None 
        2. Somerset | True | (ORG)
        3. 83 | False | None
        4. morning | False | None
        5. Grace Road | True | (LOC)
        6. Leicestershire | True | (ORG)
        7. first innings | False | None
        8. England | True | (LOC)
        9. Andy Caddick | True | (PER)
        """
    tf_exemplar_2 = """
        Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .

        Answer:
        1. Their | False | None
        2. stay | False | None
        3. title rivals | False | None
        4. Essex | True | (ORG)
        5. Derbyshire | True | (ORG)
        6. Surrey | True | (ORG)
        7. victory | False | None
        8. Kent | True | (ORG)
        9. Nottinghamshire | True | (ORG)

        """

    tf_exemplar_3 = """
        But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .

        Answer:
        1. money | False | None
        2. savings account | False | None
        3. 5.3 | False | None
        4. June | False | None
        5. July | False | None

        """
    tf_exemplars = [tf_exemplar_1, tf_exemplar_2, tf_exemplar_3]

    exemplar_1 = """
        After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
        
        Answer:
        1. Somerset | (ORG)
        2. Grace Road | (LOC)
        3. Leicestershire | (ORG). 
        4. England | (LOC)
        5. Andy Caddick | (PER) 
    """
    exemplar_2 = """
        Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
        
        Answer:
        1. Essex | (ORG)
        2. Derbyshire | (ORG)
        3. Surrey | (ORG)
        4. Kent | (ORG)
        5. Nottinghamshire | (ORG)
    """

    exemplar_3 = """
    But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .

    Answer:
    1. 
    """
    exemplars = [exemplar_1, exemplar_2, exemplar_3]

    type_exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .

    Entity Phrase: Somerset
    Answer: Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)
    
    Entity Phrase: England
    Answer: England is a country hence it is a location (LOC)
    
    Entity Phrase: Grace Road
    Answer: at Grace Road indicates this is a location or venue (LOC)    
    """

    type_exemplar_2 = """
    Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
    
    Entity Phrase: Essex
    Answer: As they are tital rivals, Essex is a sports team and not a location (ORG)
    
    Entity Phrase: Nottinghamshire
    Answer: As Nottinghamshire defeated Kent, this is a sports team not a location (ORG)
    """
    type_exemplars = [type_exemplar_1, type_exemplar_2]

    dispute_exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .

    Entity Phrase: Somerset, Options: [(LOC), (ORG)]
    Answer: Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)

    Entity Phrase: England, Options: [(LOC), (PER)]
    Answer: England is a country hence it is a location not a person (LOC)

    Entity Phrase: Grace Road, Options: [(LOC), (ORG)]
    Answer: at Grace Road indicates this is a location or venue (LOC)    
    """

    dispute_exemplar_2 = """
    Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .

    Entity Phrase: Essex, Options: [(LOC), (ORG)]
    Answer: As they are tital rivals, Essex is a sports team and not a location (ORG)

    Entity Phrase: Nottinghamshire, Options: [(LOC), (ORG)]
    Answer: As Nottinghamshire defeated Kent, this is a sports team not a location (ORG)
    """
    dispute_exemplars = [dispute_exemplar_1, dispute_exemplar_2]


class ConllConfig(Config):
    defn = (
        "An entity is a person (PER), title, named organization (ORG), location (LOC), country (LOC) or nationality (MISC)."
        "Names, first names, last names, countries are entities. Nationalities are entities even if they are "
        "adjectives. Sports, sporting events, adjectives, verbs, numbers, "
        "adverbs, abstract concepts, sports, are not entities. Dates, years and times are not entities. "
        "Possessive words like I, you, him and me are not entities. "
        "If a sporting team has the name of their location and the location is used to refer to the team, "
        "it is an entity which is an organisation, not a location"
    )

    defn = "An entity is a person (PER), title, named organization (ORG), location (LOC), country (LOC) or nationality (MISC)."

    cot_exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
    
    Answer:
    1. bowling | False | as it is an action
    2. Somerset | True | Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)
    3. 83 | False | as it is a number 
    4. morning | False| as it represents a time of day, with no distinct and independant existence
    5. Grace Road | True | the game is played at Grace Road, hence it is a place or location (LOC)
    6. Leicestershire | True | is the name of a cricket team that is based in the town of Leicestershire, hence it is an organisation (ORG). 
    7. first innings | False | as it is an abstract concept of a phase in play of cricket
    8. England | True | as it is a place or location (LOC)
    9. Andy Caddick | True | as it is the name of a person. (PER) 
    """
    cot_exemplar_2 = """
    Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
    
    Answer:
    1. Their | False | as it is a possessive pronoun
    2. stay | False | as it is an action
    3. title rivals | False | as it is an abstract concept
    4. Essex | True |  Essex are title rivals is it a sporting team organisation not a location (ORG)
    5. Derbyshire | True |  Derbyshire are title rivals is it a sporting team organisation not a location (ORG)
    6. Surrey | True |  Surrey are title rivals is it a sporting team organisation not a location (ORG)
    7. victory | False | as it is an abstract concept
    8. Kent | True |  Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
    9. Nottinghamshire | True |  Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
    
    """

    cot_exemplar_3 = """
    But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .
    
    Answer:
    1. money | False | as it is not a named person, organization or location
    2. savings account | False | as it is not a person, organization or location
    3. 5.3 | False | as it is a number
    4. June | False | as it is a date
    5. July | False | as it is a date
    """

    cot_exemplars = [cot_exemplar_1, cot_exemplar_2, cot_exemplar_3]

    no_tf_exemplar_1 = """
        After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
        
        Answer:
        1. Somerset | Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)
        2. Grace Road | the game is played at Grace Road, hence it is a place or location (LOC)
        3. Leicestershire | is the name of a cricket team that is based in the town of Leicestershire, hence it is an organisation (ORG). 
        4. England | as it is a place or location (LOC)
        5. Andy Caddick | as it is the name of a person. (PER) 
        """
    no_tf_exemplar_2 = """
        Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
        
        Answer:
        1. Essex | since Essex are title rivals is it a sporting team organisation not a location (ORG)
        2. Derbyshire | since Derbyshire are title rivals is it a sporting team organisation not a location (ORG)
        3. Surrey | since Surrey are title rivals is it a sporting team organisation not a location (ORG)
        4. Kent | since Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
        5. Nottinghamshire | since Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
        """

    no_tf_exemplar_3 = """
        But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .

        Answer:
        1. 

        """
    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2, no_tf_exemplar_3]

    tf_exemplar_1 = """
        After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .

        Answer:
        1. bowling | False | None 
        2. Somerset | True | (ORG)
        3. 83 | False | None
        4. morning | False | None
        5. Grace Road | True | (LOC)
        6. Leicestershire | True | (ORG)
        7. first innings | False | None
        8. England | True | (LOC)
        9. Andy Caddick | True | (PER)
        """
    tf_exemplar_2 = """
        Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .

        Answer:
        1. Their | False | None
        2. stay | False | None
        3. title rivals | False | None
        4. Essex | True | (ORG)
        5. Derbyshire | True | (ORG)
        6. Surrey | True | (ORG)
        7. victory | False | None
        8. Kent | True | (ORG)
        9. Nottinghamshire | True | (ORG)

        """

    tf_exemplar_3 = """
        But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .

        Answer:
        1. money | False | None
        2. savings account | False | None
        3. 5.3 | False | None
        4. June | False | None
        5. July | False | None

        """
    tf_exemplars = [tf_exemplar_1, tf_exemplar_2, tf_exemplar_3]

    exemplar_1 = """
        After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
        
        Answer:
        1. Somerset | (ORG)
        2. Grace Road | (LOC)
        3. Leicestershire | (ORG). 
        4. England | (LOC)
        5. Andy Caddick | (PER) 
    """
    exemplar_2 = """
        Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
        
        Answer:
        1. Essex | (ORG)
        2. Derbyshire | (ORG)
        3. Surrey | (ORG)
        4. Kent | (ORG)
        5. Nottinghamshire | (ORG)
    """

    exemplar_3 = """
    But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .

    Answer:
    1. 
    """
    exemplars = [exemplar_1, exemplar_2, exemplar_3]

    type_exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .

    Entity Phrase: Somerset
    Answer: Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)
    
    Entity Phrase: England
    Answer: England is a country hence it is a location (LOC)
    
    Entity Phrase: Grace Road
    Answer: at Grace Road indicates this is a location or venue (LOC)    
    """

    type_exemplar_2 = """
    Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
    
    Entity Phrase: Essex
    Answer: As they are tital rivals, Essex is a sports team and not a location (ORG)
    
    Entity Phrase: Nottinghamshire
    Answer: As Nottinghamshire defeated Kent, this is a sports team not a location (ORG)
    """
    type_exemplars = [type_exemplar_1, type_exemplar_2]

    dispute_exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .

    Entity Phrase: Somerset, Options: [(LOC), (ORG)]
    Answer: Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)

    Entity Phrase: England, Options: [(LOC), (PER)]
    Answer: England is a country hence it is a location not a person (LOC)

    Entity Phrase: Grace Road, Options: [(LOC), (ORG)]
    Answer: at Grace Road indicates this is a location or venue (LOC)    
    """

    dispute_exemplar_2 = """
    Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .

    Entity Phrase: Essex, Options: [(LOC), (ORG)]
    Answer: As they are tital rivals, Essex is a sports team and not a location (ORG)

    Entity Phrase: Nottinghamshire, Options: [(LOC), (ORG)]
    Answer: As Nottinghamshire defeated Kent, this is a sports team not a location (ORG)
    """
    dispute_exemplars = [dispute_exemplar_1, dispute_exemplar_2]
