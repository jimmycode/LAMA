# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .bert_connector import Bert
from .elmo_connector import Elmo
from .fairseq_connector import Fairseq
from .gpt_connector import GPT
from .transformerxl_connector import TransformerXL
from .xlm_connector import XLM


def build_model_by_name(lm, args, verbose=True):
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """
    MODEL_NAME_TO_CLASS = dict(
        fairseq=Fairseq,
        elmo=Elmo,
        bert=Bert,
        gpt=GPT,
        transformerxl=TransformerXL,
        xlm=XLM
    )
    if lm not in MODEL_NAME_TO_CLASS:
        raise ValueError("Unrecognized Language Model: %s." % lm)
    if verbose:
        print("Loading %s model..." % lm)
    return MODEL_NAME_TO_CLASS[lm](args)
