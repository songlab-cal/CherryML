__version__ = "v0.2.0"

from cherryml._cherryml_public_api import cherryml_public_api
from cherryml._siterm_public_api import learn_site_specific_rate_matrices
from cherryml.counting import count_co_transitions, count_transitions
from cherryml.estimation import jtt_ipw, quantized_transitions_mle
from cherryml.estimation_end_to_end import (
    coevolution_end_to_end_with_cherryml_optimizer,
    lg_end_to_end_with_cherryml_optimizer,
    lg_end_to_end_with_em_optimizer,
)
from cherryml.evaluation import compute_log_likelihoods
from cherryml.phylogeny_estimation import fast_tree, phyml
from cherryml.types import PhylogenyEstimatorType

from . import caching

caching.set_hash_len(64)
caching.set_dir_levels(0)
