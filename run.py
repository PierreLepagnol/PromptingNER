import numpy as np
import pandas as pd
from seqeval.metrics import f1_score
from tqdm import tqdm
from transformers import PreTrainedModel

from algorithms import Algorithm, AtisConfig, ConllConfig, MediaConfig
from data import load_conll2003, load_media, sample_all_types
from utils.model import load_model_and_tokenizer


def eval_dataset(val: pd.DataFrame, algorithm: Algorithm, print_every=10):
    columns = ["text", "entities", "truth", "pred", "meta", "f1"]
    data = []
    preds, truths = [], []

    for i, info in tqdm(enumerate(val.iterrows()), total=len(val)):
        index, q = info
        para = q["text"]
        entities = q["entities"]
        subdata = [para, entities, q["exact_types"]]
        algorithm.set_para(para)

        types = None
        flag = False

        while not flag:
            true_tokens = None
            if "true_tokens" in val.columns:
                true_tokens = q["true_tokens"]
            span_pred, meta = algorithm.perform_span(true_tokens=true_tokens, verbose=False)
            p = [span_pred]
            t = [q["exact_types"]]
            preds.append(span_pred)
            truths.append(q["exact_types"])
            mini_f1 = f1_score(t, p)
            subdata.extend([span_pred, meta, mini_f1])
            data.append(subdata)
            f1_micro = f1_score(truths, preds, average="micro")
            flag = True
            # try:
            #
        #
        # except IndexError:
        # flag = True

        if print_every is not None:
            if i % print_every == 0:
                f1_micro = f1_score(truths, preds, average="micro")
                f1_macro = f1_score(truths, preds, average="macro")
                print(f"Iteration {i}: micro f1: {f1_micro}, macro f1: {f1_macro}")

    f1_micro = f1_score(truths, preds, average="micro")
    f1_macro = f1_score(truths, preds, average="macro")
    print(f"Finally: micro f1: {f1_micro}, macro f1: {f1_macro}")
    df = pd.DataFrame(data=data, columns=columns)
    return f1_micro, f1_macro, df


def eval_slot_filling(val: pd.DataFrame, algorithm: Algorithm, print_every=10):
    columns = ["text", "entities", "truth", "pred", "meta", "f1"]
    data = []
    preds, truths = [], []

    for i, info in tqdm(enumerate(val.iterrows()), total=len(val)):
        index, q = info
        para = q["text"]
        entities = q["entities"]
        subdata = [para, entities, q["types"]]
        algorithm.set_para(para)

        types = None
        flag = False

        while not flag:
            true_tokens = None
            if "true_tokens" in val.columns:
                true_tokens = q["true_tokens"]
            span_pred, meta = algorithm.perform_span(true_tokens=true_tokens, verbose=False)
            p = [span_pred]
            t = [q["exact_types"]]
            preds.append(span_pred)
            truths.append(q["exact_types"])
            mini_f1 = f1_score(t, p)
            subdata.extend([span_pred, meta, mini_f1])
            data.append(subdata)
            f1_micro = f1_score(truths, preds, average="micro")
            flag = True
            # try:
            #
        #
        # except IndexError:
        # flag = True

        if print_every is not None:
            if i % print_every == 0:
                f1_micro = f1_score(truths, preds, average="micro")
                f1_macro = f1_score(truths, preds, average="macro")
                print(f"Iteration {i}: micro f1: {f1_micro}, macro f1: {f1_macro}")

    f1_micro = f1_score(truths, preds, average="micro")
    f1_macro = f1_score(truths, preds, average="macro")
    print(f"Finally: micro f1: {f1_micro}, macro f1: {f1_macro}")
    df = pd.DataFrame(data=data, columns=columns)
    return f1_micro, f1_macro, df


def complete_eval(dataset: pd.DataFrame, algorithm: Algorithm, n_runs=2, limit=None, task="intent"):
    """Compute the evaluation of a dataset using a model and algorithm

    Args:
        dataset (_type_): Dataset to evaluate
        model (PreTrainedModel): Model to use
        algorithm (_type_): Algorithm to use
        n_runs (int, optional): Number of run to estimate metrics. Defaults to 2.
        limit (_type_, optional): Limite du dataset pour les test. Defaults to None.

    Returns:
        _type_:
    """
    micros = []
    macros = []

    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}")
        if limit is not None:
            small_dataset = dataset.sample(limit)
        else:
            small_dataset = dataset
        if task == "intent":
            f1_micro, f1_macro, df = eval_dataset(small_dataset, algorithm)
        else:
            f1_micro, f1_macro, df = eval_slot_filling(small_dataset, algorithm)
        micros.append(f1_micro)
        macros.append(f1_macro)
    micros = np.array(micros)
    macros = np.array(macros)
    return micros, macros, df


def eval_conll(
    algorithm: Algorithm,
    n_runs=2,
    limit=None,
    exemplar=True,
    coT=True,
    defn=True,
    tf=True,
    autogen=True,
    **kwargs,
):
    config = ConllConfig()
    algorithm.split_phrases = False
    autogen = False
    if autogen:
        algorithm.set_model_fn(model)
        conll_train = load_conll2003("train")
        subsample = sample_all_types(conll_train, 3)
        texts = subsample["text"].tolist()
        tokens = subsample["text"].apply(lambda x: x.split(" ")).tolist()
        labels = subsample["exact_types"].tolist()
        config.autogenerate_annotations(algorithm, texts, tokens, labels)

    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    conll = load_conll2003("test")
    return complete_eval(conll, algorithm, n_runs=n_runs, limit=limit)


def eval_atis(
    model: PreTrainedModel,
    algorithm: Algorithm,
    n_runs=2,
    limit=None,
    exemplar=True,
    coT=True,
    defn=True,
    tf=True,
    autogen=True,
    **kwargs,
):
    config = AtisConfig()
    return False


def eval_media(
    algorithm: Algorithm,
    n_runs=2,
    limit=None,
    exemplar=True,
    coT=True,
    defn=True,
    tf=True,
    generate_labels_from_lm=False,
    **kwargs,
):
    config = MediaConfig()
    algorithm.split_phrases = False

    if generate_labels_from_lm:
        dataset_train = load_media(split="train", version=kwargs["version"])
        subsample = sample_all_types(dataset_train, 3)
        texts = subsample["text"].tolist()
        tokens = subsample["text"].apply(lambda x: x.split(" ")).tolist()
        labels = subsample["exact_types"].tolist()
        config.autogenerate_annotations(algorithm, texts, tokens, labels)

    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    dataset = load_media(split="test", version=kwargs["version"])
    return complete_eval(dataset, algorithm, n_runs=n_runs, limit=limit, task="slot_filling")


def run(dataset="conll", exemplar=True, coT=True, defn=True, tf=True, name_meta=""):
    print(f"Running for: {dataset}")
    model, tokenizer = load_model_and_tokenizer("bigscience/bloomz-560m", model_type="causal")
    algorithm = Algorithm(model=model, tokenizer=tokenizer)

    # Constants - Define outside the function if these don't change across calls
    other_limit = 3
    other_nruns = 1
    subdataset = ""

    # Dataset evaluation function mapping
    dataset_to_evalfn = {"conll": eval_conll, "atis": eval_atis, "media": eval_media}

    # Check if the dataset is supported
    eval_fn = dataset_to_evalfn.get(dataset)
    if not eval_fn:
        raise ValueError(f"Unknown Dataset: {dataset}")

    # Run the evaluation
    micros, macros, df = eval_fn(
        algorithm, n_runs=other_nruns, limit=other_limit, coT=coT, defn=defn, tf=tf, version="original"
    )

    # print(
    #     f"Final Results For {name_meta} | {dataset} {'('+subdataset+')' if subdataset is not None else ''}) "
    #     f"|CoT {coT} | Exemplar {exemplar} (tf {tf}) |Defn {defn}"
    # )
    print(f"Micro f1_means: {micros.mean()}")
    print(f"Micro f1_stds: {micros.std()}")
    print(f"Macro f1_means: {macros.mean()}")
    print(f"Macro f1_stds: {macros.std()}")
    save_path = f"results/{name_meta}{dataset}{subdataset}.csv"
    df.to_csv(save_path, index=False)
    return micros, macros


def run_all_datasets(
    exemplar=True,
    coT=True,
    defn=True,
    tf=True,
    name_meta="",
    dataset_exclude=[],
    subdataset_exclude=[],
):
    d = {}
    # datasets = ["conll", "genia", "crossner", "fewnerd", "tweetner", "fabner"]
    # subdatasets = {"crossner": ["politics", "literature", "ai", "science", "music"], "fewnerd": ["test"]}
    datasets = ["media", "atis", "snips", "conll"]
    for dataset in datasets:
        micro, macro = run(dataset=dataset, coT=coT, exemplar=exemplar, defn=defn, tf=tf, name_meta=name_meta)
        d[dataset] = [(macro * 100).mean(), (macro * 100).std(), (micro * 100).mean(), (micro * 100).std()]
        if dataset in dataset_exclude:
            continue
    return d


# else:
# for s in sub:
#     if s in subdataset_exclude:
#         continue
#     macro, micro = run(
#         dataset=dataset,
#         subdataset=s,
#         coT=coT,
#         exemplar=exemplar,
#         defn=defn,
#         tf=tf,
#         name_meta=name_meta,
#     )
#     d[f"{dataset}_{s}"] = [
# (macro * 100).mean(),
# (macro * 100).std(),
# (micro * 100).mean(),
# (micro * 100).std(),
# ]


def ablate_all(
    gpt=False,
    vary_cot=True,
    vary_exemplar=True,
    vary_tf=True,
    vary_defn=True,
    dataset_exclude=["genia"],
    subdataset_exclude=[],
):
    cot_options = [True, False] if vary_cot else [True]
    exemplar_options = [True, False] if vary_exemplar else [True]
    tf_options = [True, False] if vary_tf else [True]
    defn_options = [True, False] if vary_defn else [True]
    # first take off cot then tf then example then defn
    res_d = {}
    for defn in defn_options:
        for exemplar in exemplar_options:
            for cot in cot_options:
                for tf in tf_options:
                    key = (defn, exemplar, cot, tf)
                    res_d[key] = run_all_datasets(
                        gpt=gpt,
                        exemplar=exemplar,
                        coT=cot,
                        defn=defn,
                        tf=tf,
                        dataset_exclude=dataset_exclude,
                        subdataset_exclude=subdataset_exclude,
                    )

    print(
        f"Ablations Done.... \nFinal Results For All: f1 Macro Mean, f1 Macro Std, f1 Micro Mean, f1 Micro Std"
    )
    for defn in defn_options:
        for exemplar in exemplar_options:
            for cot in cot_options:
                for tf in tf_options:
                    key = (defn, exemplar, cot, tf)
                    print(f"Defn: {key[0]}\tExemplar: {key[1]}\tCoT: {key[2]}\ttf:{key[3]}")
                    for dataset_key in res_d[key]:
                        print(f"\t{dataset_key}")
                        formatted = [f"{i:.3f}" for i in res_d[key][dataset_key]]
                        print(f"\t\t{formatted}")
    return


def ablate_best(
    gpt=False, dataset_exclude=["genia"], subdataset_exclude=["politics", "literature", "train", "dev"]
):
    configurations = [
        (True, True, True, True),
        (False, True, True, True),
        (True, False, True, True),
        (True, True, False, True),
        (True, True, True, False),
    ]
    res_d = {}
    for defn, exemplar, cot, tf in configurations:
        key = (defn, exemplar, cot, tf)
        res_d[key] = run_all_datasets(
            gpt=gpt,
            exemplar=exemplar,
            coT=cot,
            defn=defn,
            tf=tf,
            dataset_exclude=dataset_exclude,
            subdataset_exclude=subdataset_exclude,
        )

    print(
        f"Ablations Done.... \nFinal Results For All: f1 Macro Mean, f1 Macro Std, f1 Micro Mean, f1 Micro Std"
    )
    for defn, exemplar, cot, tf in configurations:
        key = (defn, exemplar, cot, tf)
        print(f"Defn: {key[0]}\tExemplar: {key[1]}\tCoT: {key[2]}\ttf:{key[3]}")
        for dataset_key in res_d[key]:
            print(f"\t{dataset_key}")
            formatted = [f"{i:.3f}" for i in res_d[key][dataset_key]]
            print(f"\t\t{formatted}")
    return


if __name__ == "__main__":
    run_all_datasets()
