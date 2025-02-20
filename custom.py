from algorithms import Algorithm, Config
from models import OpenAIGPT


class ExampleConfig(Config):
    defn = "An entity is a person (person), title, named organization (org), location (loc), country (loc) or nationality (misc)." \
           "Names, first names, last names, countries are entities. Nationalities are entities even if they are " \
           "adjectives. Sports, sporting events, adjectives, verbs, numbers, " \
           "adverbs, abstract concepts, sports, are not entities. Dates, years and times are not entities. " \
           "Possessive words like I, you, him and me are not entities. " \
           "If a sporting team has the name of their location and the location is used to refer to the team, " \
           "it is an entity which is an organisation, not a location"

    cot_exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .

    Answer:
    1. bowling | False | as it is an action
    2. Somerset | True | Somerset is used as a sporting team here, not a location hence it is an organisation (org)
    3. 83 | False | as it is a number 
    4. morning | False| as it represents a time of day, with no distinct and independant existence
    5. Grace Road | True | the game is played at Grace Road, hence it is a place or location (loc)
    6. Leicestershire | True | is the name of a cricket team that is based in the town of Leicestershire, hence it is an organisation (org). 
    7. first innings | False | as it is an abstract concept of a phase in play of cricket
    8. England | True | as it is a place or location (loc)
    9. Andy Caddick | True | as it is the name of a person. (person) 
    """
    cot_exemplar_2 = """
    Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .

    Answer:
    1. Their | False | as it is a possessive pronoun
    2. stay | False | as it is an action
    3. title rivals | False | as it is an abstract concept
    4. Essex | True |  Essex are title rivals is it a sporting team organisation not a location (org)
    5. Derbyshire | True |  Derbyshire are title rivals is it a sporting team organisation not a location (org)
    6. Surrey | True |  Surrey are title rivals is it a sporting team organisation not a location (org)
    7. victory | False | as it is an abstract concept
    8. Kent | True |  Kent lost to Nottinghamshire, it is a sporting team organisation not a location (org)
    9. Nottinghamshire | True |  Kent lost to Nottinghamshire, it is a sporting team organisation not a location (org)

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


# Fill out this custom config class to try out a new configuration,
# you can refer to the example config class to see how to use it
class CustomConfig(Config):
    defn = """
    Describe the specifications for what counts as an entity here, 
    mention groups of words that are entities and groups of words that are not
    """

    # Make examples of sentences and the task being performed in the following format
    cot_exemplar_1 = """
    Sentence from which entities must be extracted
    
    Answer:
    1. Candidate | True | reason why the candidate should be considered an entity (entity_type)
    2. Candidate | False | reason why the candidate should not be considered an entity (entity_type)
    .....
    """

    # You can add as many exemplars as you want and add them to this list, just do not change the name of the list
    cot_exemplars = [cot_exemplar_1]


# This returns the NER system that will use the OpenAI GPT model as specified in models.py (lines 12-14),
# change this to use a different model
def get_ner_system(split_phrases=False, identify_types=True, verbose=True):  # Set split_phrases with true to automatically split all identified phrases
    algorithm = Algorithm(model_fn=OpenAIGPT.query, split_phrases=split_phrases, identify_types=identify_types)

    def get_entities(sentence: str, verbose=verbose):  #
        """

        :param sentence: string input sentence
        :param verbose: boolean True will print model output
        :return: <entities, entity_types, metadata> if identify_types is true, else <entities, metadata>
        """
        algorithm.para = sentence
        return algorithm.perform(verbose=verbose)
    return get_entities  # You can call this function with a sentence input
