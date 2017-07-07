model=MSSG
dim=300D
method=compress

python eliminator.py \
    --global-model ../../data/ijcnlp-vectors/wiki_vectors.$model.$dim.6K_global.vec \
    --local-model ../../data/ijcnlp-vectors/wiki_vectors.$model.$dim.6K_local.vec \
    --context-model ../../data/ijcnlp-vectors/wiki_vectors.$model.$dim.6K_context.vec \
    --store-path ../../data/ijcnlp-vectors/wiki_vectors.$model.$dim.6K_local.transform.$method \
    --pseudo-threshold 0.5 \
    --training-method $method
