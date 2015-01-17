#!/usr/bin/env python
import optparse, sys, models, itertools, copy, math, time, os, gzip
from collections import namedtuple, defaultdict

class hypothesis(namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, coverage, end, future_logprob, adjusted_logprob")):
    __slots__ = ()

def extract_data(path):
    return gzip.open(path, 'r') if path[-3:] == '.gz' else open(path, 'r')

def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

def extract_tm_logprob(h):
    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)

def extract_features(h, feature_count):
    if h.predecessor is None:
        return [0.0 for _ in xrange(feature_count)]
    else:
        if h.phrase.features == []:
            return [x + y for x, y in zip([0.0 for _ in xrange(feature_count)], extract_features(h.predecessor, feature_count))]
        else:
            return [x + y for x, y in zip(h.phrase.features, extract_features(h.predecessor, feature_count))]

def print_features(features):
    return " ".join(str(f) for f in features)

# tm should translate unknown words as-is with probability 1
def initialize(french, tm):
    for word in set(sum(french,())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, 0.0, [])]

def precompute_spans(f, tm, lm, f_len):
    spans = defaultdict(float)
    for span in xrange(1, f_len + 1):
        for i in xrange(0, f_len - span - 1):
            for j in xrange(i + 1, f_len + 1):
                spans[(i, j)] = float("-inf")
                for k in xrange(i + 1, j):
                    spans[(i, j)] = max(spans[(i, k)] + spans[(k, j)], spans[(i, j)])
                candidates = tm.get(f[i:j], [])
                if candidates == []:
                    continue
                # find the best lm and tm
                for c in candidates:
                    logprob = c.logprob
                    lm_state = tuple()
                    for word in c.english.split():
                        (lm_state, word_logprob) = lm.score(lm_state, word)
                        logprob += word_logprob
                    spans[(i, j)] = max(spans[(i, j)], logprob)
    return spans

def estimate_future_logprob(coverage, spans, f_len):
    future_logprob = 0.0
    inside = False
    start = -1
    for i in xrange(f_len):
        if coverage[i] == 1 and inside:
            future_logprob += spans[(start, i)]
            inside = False
        elif coverage[i] == 0 and not inside:
            inside = True
            start = i
        if inside and (i == f_len):
            future_logprob += spans[(start, f_len)]
    return future_logprob

def get_phrases(h, f, tm, f_len, distortion_limit):
    untranslated = (x for (x, v) in enumerate(h.coverage) if v == 0)
    #candidates = itertools.ifilter(lambda x: abs(h.end - x) <= distortion_limit, untranslated)
    for start in untranslated:
        #for j in xrange(start, start + 1 + distortion_limit):
        for j in xrange(start, f_len):
            for k in xrange(j+1, f_len+1):
                #sys.stderr.write("%s\n" % (h.coverage))
                #sys.stderr.write("j: %s k: %s end: %s\n" % (j,k,h.end))
                translated = h.coverage[j:k].count(1)
                if translated == 0:
                    words = f[j:k]
                    if words in tm:
                        yield tm[words], j, k

# put beam width back in
def decode(french, tm, lm, stack_max, distortion_limit, distortion_penalty, offset, feature_count, verbose, dump, ignore):
    sys.stderr.write("Decoding %s...\n" % opts.input)
    for (s, f) in enumerate(french):
        f_len = len(f)
        spans = precompute_spans(f, tm, lm, f_len) # get the FCE table
        coverage = [0 for _ in f]
        initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, coverage, 0, 0.0, 0.0)
        stacks = [{} for _ in xrange(f_len + 1)]
        stacks[0][(0, lm.begin(), tuple(coverage))] = initial_hypothesis
        for i, stack in enumerate(stacks[:-1]):
            #sys.stderr.write("### on stack %s, size is %s ###\n" % (i, len(stack)))
            for h in sorted(stack.itervalues(),key=lambda h: -h.adjusted_logprob)[:stack_max]: # prune
                #sys.stderr.write("stack: %s\n" % str(h))
                for (phrases, j, k) in get_phrases(h, f, tm, f_len, distortion_limit):
                    for phrase in phrases:
                        #sys.stderr.write("saw j: %s k: %s\n" % (j,k))
                        if abs(h.end  - j) > distortion_limit:
                            continue
                        #sys.stderr.write("kept j: %s k: %s\n" % (j,k))
                        logprob = h.logprob + phrase.logprob
                        lm_state = h.lm_state
                        for word in phrase.english.split():
                            (lm_state, word_logprob) = lm.score(lm_state, word)
                            logprob += word_logprob
                        coverage = copy.deepcopy(h.coverage) # deep copy of current coverage
                        for c in xrange(j, k):
                            coverage[c] = 1
                        translated = [x for (x, v) in enumerate(coverage) if v == 1]
                        end = translated[-1]
                        #end = k
                        covered = coverage.count(1)
                        logprob += lm.end(lm_state) if covered == f_len else 0.0
                        future_logprob = estimate_future_logprob(coverage, spans, f_len)
                        #penalty = math.log10(math.pow(distortion_penalty, abs(h.end - j)))
                        penalty = 0.0
                        adjusted_logprob = future_logprob + penalty + logprob - h.logprob
                        if h.adjusted_logprob != 0.0: # might not need this check
                            adjusted_logprob += h.adjusted_logprob - h.future_logprob
                        new_hypothesis = hypothesis(logprob, lm_state, h, phrase, coverage, end, future_logprob, adjusted_logprob)
                        key = (end, lm_state, tuple(coverage))
                        if key not in stacks[covered] or stacks[covered][key].logprob < logprob:
                            stacks[covered][key] = new_hypothesis
        assert(len(stacks[-1]) > 0)
        if dump:
            for winner in sorted(stacks[-1].itervalues(),key=lambda h: -h.logprob)[:stack_max]:
                #sys.stderr.write("%s\n" % str(winner))
                assert(winner.coverage.count(1) == len(winner.coverage))
                tm_logprob = extract_tm_logprob(winner)
                lm_logprob = winner.logprob - tm_logprob
                if not ignore:
                    assert(tm_logprob <= 0.0)
                    assert(lm_logprob <= 0.0)
                print s+offset, "|||", extract_english(winner), "|||", lm_logprob, print_features(extract_features(winner, feature_count))
            if verbose:
                sys.stderr.write("s = %s, finished dumping nbest for sentence\n" % (s+offset))
        else:
            winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
            assert(winner.coverage.count(1) == len(winner.coverage))
            print extract_english(winner)
            if verbose:
                tm_logprob = extract_tm_logprob(winner)
                lm_logprob = winner.logprob - tm_logprob
                if not ignore:
                    assert(tm_logprob <= 0.0)
                    assert(lm_logprob <= 0.0)
                sys.stderr.write("s = %s, LM = %f, TM = %f, Total = %f\n" % (s+offset, lm_logprob, tm_logprob, winner.logprob))

# change these to os.path.join("data", "test.en")
if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--input", dest="input", default="../toy/train.cn", help="File containing sentences to translate (default=../toy/train.cn)")
    optparser.add_option("-t", "--translation-model", dest="tm", default="../toy/phrase-table/phrase_table.out", help="File containing translation model (default=../toy/phrase-table/phrase_table.out)")
    optparser.add_option("-l", "--language-model", dest="lm", default="../lm/en.tiny.3g.arpa", help="File containing ARPA-format language model (default=../lm/en.tiny.3g.arpa)")
    optparser.add_option("-w", "--weights", dest="weights", default="./default.weights", help="File containing weights (default=../default.weights)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
    optparser.add_option("-k", "--translations-per-phrase", dest="k", default=5, type="int", help="Limit on number of translations to consider per phrase (default=5)")
    optparser.add_option("-s", "--stack-size", dest="s", default=100, type="int", help="Maximum stack size (default=100)")
    optparser.add_option("-d", "--distortion-limit", dest="d", default=1, type="int", help="Maximum distortion limit (default=1)")
    optparser.add_option("-p", "--distortion-penalty", dest="p", default=0.9, type="float", help="Distortion penalty (default=0.9)")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Verbose mode (default=off)")
    optparser.add_option("-a", "--dump-all", dest="dump", action="store_true", default=False, help="Output the entire stack instead of best sentences (default=off)")
    optparser.add_option("-r", "--ignore-positive-scores", dest="ignore", action="store_true", default=False, help="Ignore when log scores are positive (default=off)")
    optparser.add_option("-o", "--sentence-offset", dest="offset", default=0, type="int", help="Sentence offset used when input file is chunked (default=0)")
    opts = optparser.parse_args()[0]
    if opts.s < 1:
        sys.stderr.write("Stack size should be at least 1.\n")
        exit(0)
    if opts.d < 1:
        sys.stderr.write("Distortion limit should be at least 1.\n")
        exit(0)
    if opts.p < 0:
        if not (0.0 <= opts.p <= 1.0):
            sys.stderr.write("Distortion penalty should be in range [0, 1].\n")
            exit(0)
    start_time = time.time()
    weight_data = extract_data(opts.weights)
    weights = [float(line.strip()) for line in weight_data]
    #if weights.count(0.2) != len(weights):
    #    weights = [weight * 0.2 for weight in weights]
    tm, feature_count = models.TM(opts.tm, opts.k, weights)
    sys.stderr.write("TM size %s\n" % len(tm))
    lm = models.LM(opts.lm)
    french_data = extract_data(opts.input)
    french = [tuple(line.strip().split()) for line in french_data.readlines()[:opts.num_sents]]
    initialize(french, tm)
    decode(french, tm, lm, opts.s, opts.d, opts.p, opts.offset, feature_count, opts.verbose, opts.dump, opts.ignore)
    sys.stderr.write("\nTook %s seconds\n" % (time.time() - start_time))


