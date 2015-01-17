#!/usr/bin/env python
import optparse, sys, os
import bleu

# ../toy/train.cn
# os.path.join("data", "test.en")
optparser = optparse.OptionParser()
#optparser.add_option("-r", "--reference", dest="reference", default="../toy/train.en", help="English reference sentences")
optparser.add_option("-r", "--reference", dest="reference", default="../toy/train.en", help="English reference sentences")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
(opts,_) = optparser.parse_args()

stats = [0 for i in xrange(10)]
system = [line.strip().split() for line in sys.stdin]

# make this better
references = opts.reference.split(",")
sys.stderr.write("Using %s reference(s)...\n" % len(references))
if len(references) == 1:
    ref = [line.strip().split() for line in open(references[0])][:opts.num_sents]
    for (r,s) in zip(ref, system):
      stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(s,r))]
    print bleu.bleu(stats), bleu.smoothed_bleu(stats)
else: # what if we don't want to use all 4?
    ref1 = [line.strip().split() for line in open(references[0])][:opts.num_sents]
    ref2 = [line.strip().split() for line in open(references[1])][:opts.num_sents]
    ref3 = [line.strip().split() for line in open(references[2])][:opts.num_sents]
    ref4 = [line.strip().split() for line in open(references[3])][:opts.num_sents]
    for (r1, r2, r3, r4, s) in zip(ref1, ref2, ref3, ref4, system):
      stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(s,r1,r2,r3,r4))]
    print bleu.bleu(stats), bleu.smoothed_bleu(stats)
