#!/bin/sh

python analogy.py ../../../data/vectors/neelakantan_vectors.NP-MSSG.50D.6K_global.vec \
 ./log/neelakantan_vectors.NP-MSSG.50D.6K_local.log \
 ../../../data/vectors/neelakantan_vectors.NP-MSSG.50D.6K_local.vec \
 > ./log/neelakantan_vectors.NP-MSSG.50D.6K_local.result

python analogy.py ../../../data/vectors/neelakantan_vectors.MSSG.50D.6K_global.vec \
 ./log/neelakantan_vectors.MSSG.50D.6K_local.transform.pairwise.log \
 ../../../data/ijcnlp-vectors/neelakantan_vectors.MSSG.50D.6K_local.transform.pairwise.vec \
 > ./log/neelakantan_vectors.MSSG.50D.6K_local.transform.pairwise.result

python analogy.py ../../../data/vectors/neelakantan_vectors.MSSG.50D.6K_global.vec \
 ./log/neelakantan_vectors.MSSG.50D.6K_local.transform.compress.log \
 ../../../data/ijcnlp-vectors/neelakantan_vectors.MSSG.50D.6K_local.transform.compress.vec \
 > ./log/neelakantan_vectors.MSSG.50D.6K_local.transform.compress.result

python analogy.py ../../../data/vectors/neelakantan_vectors.MSSG.300D.6K_global.vec \
 ./log/neelakantan_vectors.MSSG.300D.6K_local.transform.pairwise.log \
 ../../../data/ijcnlp-vectors/neelakantan_vectors.MSSG.300D.6K_local.transform.pairwise.vec \
 > ./log/neelakantan_vectors.MSSG.300D.6K_local.transform.pairwise.result

python analogy.py ../../../data/vectors/neelakantan_vectors.MSSG.300D.6K_global.vec \
 ./log/neelakantan_vectors.MSSG.300D.6K_local.transform.compress.log \
 ../../../data/ijcnlp-vectors/neelakantan_vectors.MSSG.300D.6K_local.transform.compress.vec \
 > ./log/neelakantan_vectors.MSSG.300D.6K_local.transform.compress.result

python analogy.py ../../../data/vectors/neelakantan_vectors.NP-MSSG.50D.6K_global.vec \
 ./log/neelakantan_vectors.NP-MSSG.50D.6K_local.transform.pairwise.log \
 ../../../data/ijcnlp-vectors/neelakantan_vectors.NP-MSSG.50D.6K_local.transform.pairwise.vec \
 > ./log/neelakantan_vectors.NP-MSSG.50D.6K_local.transform.pairwise.result

python analogy.py ../../../data/vectors/neelakantan_vectors.NP-MSSG.50D.6K_global.vec \
 ./log/neelakantan_vectors.NP-MSSG.50D.6K_local.transform.compress.log \
 ../../../data/ijcnlp-vectors/neelakantan_vectors.NP-MSSG.50D.6K_local.transform.compress.vec \
 > ./log/neelakantan_vectors.NP-MSSG.50D.6K_local.transform.compress.result

python analogy.py ../../../data/vectors/neelakantan_vectors.NP-MSSG.300D.6K_global.vec \
 ./log/neelakantan_vectors.NP-MSSG.300D.6K_local.transform.pairwise.log \
 ../../../data/ijcnlp-vectors/neelakantan_vectors.NP-MSSG.300D.6K_local.transform.pairwise.vec \
 > ./log/neelakantan_vectors.NP-MSSG.300D.6K_local.transform.pairwise.result

python analogy.py ../../../data/vectors/neelakantan_vectors.NP-MSSG.300D.6K_global.vec \
 ./log/neelakantan_vectors.NP-MSSG.300D.6K_local.transform.compress.log \
 ../../../data/ijcnlp-vectors/neelakantan_vectors.NP-MSSG.300D.6K_local.transform.compress.vec \
 > ./log/neelakantan_vectors.NP-MSSG.300D.6K_local.transform.compress.result
