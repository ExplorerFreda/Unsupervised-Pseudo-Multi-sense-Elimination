director=ijcnlp-vectors
vectortype=wiki_vectors
model=MSSG
dim=50D
method=compress

python corpus_tagger.py \
    --global-model ../../data/$director/$vectortype.$model.$dim.6K_global.vec \
    --local-model ../../data/$director/$vectortype.$model.$dim.6K_local.vec \
    --context-model ../../data/$director/$vectortype.$model.$dim.6K_context.vec \
    --input-corpus ../../data/wiki/wiki.en.corpus \
    --output-corpus ../../data/wiki/wiki.en.corpus.wiki.tagged