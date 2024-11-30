import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP VP Conj NP VP | NP VP Conj VP
NP -> Det N | Det Adj N | N | NP P NP | NP Conj NP | Det N P NP
VP -> V | V NP | V Adv | VP Adv | VP NP
AP -> Adj | AP Adj
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()
    else:
        s = input("Sentence: ")

    s = preprocess(s)

    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return

    if not trees:
        print("Could not parse sentence.")
        return

    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [word.lower() for word in tokens if any(char.isalpha() for char in word)]


def np_chunk(tree):
    chunks = []
    ptree = nltk.tree.ParentedTree.convert(tree)

    for subtree in ptree.subtrees(lambda t: t.label() == 'NP'):
        if not any(child.label() == 'NP' for child in subtree.subtrees(lambda t: t != subtree)):
            chunks.append(subtree)

    return chunks  


if __name__ == "__main__":
    main()
