from cherryml.config import Config, sanity_check_config
from functools import partial
from ._fast_tree import fast_tree
from ._fast_cherries import fast_cherries
from ._phyml import phyml
from ._gt_tree_estimator import gt_tree_estimator
from cherryml.types import PhylogenyEstimatorType
def get_phylogeny_estimator_from_config(
    config:Config,
    gt_tree_dir: str = "",
    gt_site_rates_dir: str = "",
    gt_likelihood_dir: str = "",) -> PhylogenyEstimatorType:
    sanity_check_config(config)
    name, args = config
    if name == "fast_tree":
        return partial(fast_tree, **dict(args))
    elif name == "PhyML":
        return partial(phyml, **dict(args))
    elif name == "gt":
        return partial(gt_tree_estimator, 
                       gt_tree_dir=gt_tree_dir, 
                       gt_site_rates_dir=gt_site_rates_dir,
                       gt_likelihood_dir=gt_likelihood_dir,
                       **dict(args))
    elif name == "fast_cherries":
        return partial(fast_cherries, **dict(args))
    else:
        raise NameError(f'{name} is not a valid phylogeny estimator! Valid estimators are ["fast_tree", "PhyML", "gt"].')
    