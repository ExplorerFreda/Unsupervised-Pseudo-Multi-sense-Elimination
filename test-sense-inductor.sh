model=NP-MSSG
dim=300D

python sense_inductor.py \
    --global-model ../../data/vectors/neelakantan_vectors.$model.$dim.6K_global.vec \
    --local-model ../../data/vectors/neelakantan_vectors.$model.$dim.6K_local.vec \
    --context-model ../../data/vectors/neelakantan_vectors.$model.$dim.6K_context.vec \
    --input-path ../../data/WSI/semeval2007/key/data/English_sense_induction.xml \
    --output-path ../../data/WSI/semeval2007/key/keys/pseudo/baseline.key \
    --combine-pseudo false
