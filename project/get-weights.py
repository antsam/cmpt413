#!/usr/bin/env python
import optparse, sys, os, bleu, random, math, string, copy, itertools
from collections import namedtuple, defaultdict

class candidate(namedtuple("candidate", "features, score, smoothed_bleu")):
    __slots__ = ()

def count_possible_untranslated(src, sentence):
    filtered = itertools.ifilter(lambda h: not any(c.isdigit() for c in h) , sentence) # strip numerals and words containing them
    possible_untranslated = -1.0 * (len(set(src).intersection(filtered)) + 1) # add 1 as -1 is our best score
    return possible_untranslated

# think about RARE words http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf

# nb: ensure that all features we add score at least -1 and not 0!
def get_candidates(nbest, target, source):
    ref = [line.strip().split() for line in open(target).readlines()]
    src = [line.strip().split() for line in open(source).readlines()]
    candidates = [line.strip().split("|||") for line in open(nbest).readlines()]
    nbests = [[] for _ in ref]
    original_feature_count = 0
    sys.stderr.write("Calculating smoothed bleu for n-best...")
    for n, (i, sentence, features) in enumerate(candidates):
        (i, sentence, features) = (int(i), sentence.strip(), [float(f) for f in features.strip().split()])
        # original features are: LMScore, ReorderingScore, p(f|e), lex(f|e), p(e|f), lex(e|f)
        if original_feature_count == 0:
            original_feature_count = len(features)
        sentence_split = sentence.strip().split()
        stats = tuple(bleu.bleu_stats(sentence_split, ref[i]))
        features.append(count_possible_untranslated(src[i], sentence_split))
        score = sum(features)
        nbests[i].append(candidate(features, score, bleu.smoothed_bleu(stats)))
        if n % 2000 == 0:
            sys.stderr.write(".")
    sys.stderr.write("\n")
    return nbests, original_feature_count

def assign_rank_scores(nbests):
    adjusted = []
    sys.stderr.write("Checking order of bleu and scores...")
    for (k, nbest) in enumerate(nbests):
        current = []
        # sort by descending bleu scores and compare the model scores
        bleu_sorted = sorted(nbest, key=lambda c: -c.smoothed_bleu)
        lm_sorted = sorted(nbest, key=lambda c: -c.score)
        for i in xrange(len(bleu_sorted)):
            tmp = copy.deepcopy(lm_sorted[i])
            if bleu_sorted[i] == lm_sorted[i]:
                tmp.features.append(-1.0)
            else:
                tmp.features.append(-1.0 * (math.fabs((i+1) - (bleu_sorted.index(lm_sorted[i]) + 1)) + 1))
            current.append(tmp)
        adjusted.append(current) # maybe shuffle current before appending it?
        if k % 50 == 0:
            sys.stderr.write(".")
    sys.stderr.write("\n")
    return adjusted

# Save some time by not caring if s1 > s2 since they're random choices.
# Only care if one of them is bigger than the other!
def get_samples(nbest, tau, alpha):
    random.seed()
    for _ in xrange(tau):
        s1 = random.choice(nbest)
        s2 = random.choice(nbest)
        if (s1 != s2) and (math.fabs(s1.smoothed_bleu - s2.smoothed_bleu) > alpha):
            yield (s1, s2) if s1.smoothed_bleu > s2.smoothed_bleu else (s2, s1)

# dot product
def dot(x, y):
    return sum([(x_i * y_i) for (x_i, y_i) in zip(x, y)])

def perceptron(nbests, epochs, tau, eta, xi, alpha):
    theta = [0.0 for _ in xrange(len(nbests[0][0].features))] # so things don't explode
    for e in xrange(epochs):
        mistakes = 0.0
        observed = 0.0
        sys.stderr.write("Starting iteration %s..." % (e+1))
        for nbest in nbests:
            for (s1, s2) in sorted(get_samples(nbest, tau, alpha), key=lambda (s1, s2): s2.smoothed_bleu - s1.smoothed_bleu)[:xi]:
                # theta * s1.features <= theta * s2.features
                # (theta * s1.features) - (theta * s2.features) <= 0
                # theta * (s1.features - s2.features) <= 0
                s = [(s1_j - s2_j) for (s1_j, s2_j) in zip(s1.features, s2.features)]
                if dot(s, theta) <= 0:
                    mistakes += 1
                    theta = [((eta * s_j) + t_j) for (s_j, t_j) in zip(s, theta)] # theta += eta * (s1.features - s2.features)
                observed += 1
                if observed % 2000 == 0:
                    sys.stderr.write(".")
        sys.stderr.write("\n")
        sys.stderr.write("Errors: %d, Error rate: %f (%d observed)\n" % (mistakes, float(mistakes)/observed, observed))
        theta = [t/(observed) for t in theta]
    return theta

def output(theta, limit):
    trimmed = theta[:limit]
    print "\n".join([str(t) for t in trimmed])

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "train.nbest"), help="N-best file")
    optparser.add_option("-e", "--target", dest="target", default=os.path.join("data", "train.en"), help="Target file")
    optparser.add_option("-f", "--source", dest="source", default=os.path.join("data", "train.fr"), help="Source file")
    optparser.add_option("-t", "--tau", dest="tau", default=5000, type="int", help="Samples generated from n-best list per input sentence (default=5000)")
    optparser.add_option("-a", "--alpha", dest="alpha", default=0.21, type="float", help="Sampler acceptance cutoff (default=0.21)")
    optparser.add_option("-x", "--xi", dest="xi", default=100, type="int", help="Training data generated from the samples tau (default=100)")
    optparser.add_option("-r", "--eta", dest="eta", default=0.1, type="float", help="Perceptron learning rate (default=0.1)")
    optparser.add_option("-i", "--epochs", dest="epochs", default=5, type="int", help="Number of epochs for perceptron training (default=5)")
    (opts, _) = optparser.parse_args()
    (nbests, original_feature_count) = get_candidates(opts.nbest, opts.target, opts.source)
    nbests = assign_rank_scores(nbests)
    theta = perceptron(nbests, opts.epochs, opts.tau, opts.eta, opts.xi, opts.alpha)
    output(theta, original_feature_count)

