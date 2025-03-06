from cherryml.config import create_config_from_dict

from typing import List
config_list = {
    "CherryML with FastCherries":{
        "config":create_config_from_dict({
            "identifier":"fast_cherries", 
            "args":{"num_rate_categories":4, "max_iters":50}}
        ),
        "color":"red",
        "style":"o-",
    },   
    "CherryML with FastTree": {
        "config":create_config_from_dict(
            {
                "identifier":"fast_tree",
                "args":{"num_rate_categories":4}
            }
        ), 
        "color":"blue",
        "style":"o-",
    },
    "FastCherries":{
        "config":create_config_from_dict({
            "identifier":"fast_cherries", 
            "args":{"num_rate_categories":4, "max_iters":50}}
        ),
        "color":"red",
        "style":"o-",
    },   
    "FastTree": {
        "config":create_config_from_dict(
            {
                "identifier":"fast_tree",
                "args":{"num_rate_categories":4}
            }
        ), 
        "color":"blue",
        "style":"o-",
    },
    "CherryML with true trees": {
        "config":create_config_from_dict(
            {
                "identifier":"gt",
                "args":{"num_rate_categories":1}
            }
        ), 
        "color":"black",
        "style":"o--",
    },
}

def get_configs_and_styles_from_name(labels:List[str]):
    configs = [] 
    styles = [] 
    colors = []
    for label in labels:
        entry = config_list[label]
        configs.append(entry["config"])
        styles.append(entry["style"])
        colors.append(entry["color"])
    return configs, styles, colors
