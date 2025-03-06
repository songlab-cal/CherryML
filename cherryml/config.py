from typing import Any, Dict, List, Tuple

# A config is just the class name (or function name) followed by the list of
# arguments.
Config = Tuple[str, List[Tuple[str, Any]]]


def create_config_from_dict(config_dict: Dict) -> Config:
    if sorted(list(config_dict.keys())) != ["args", "identifier"]:
        raise ValueError('config_dict should have keys ["args", "identifier"]')
    identifier = config_dict["identifier"]
    args_dict = config_dict["args"]
    config = (identifier, sorted(args_dict.items()))
    return config


def sanity_check_config(config: Config):
    print(config)
    identifier, args = config
    for i in range(len(args) - 1):
        if args[i][0] >= args[i + 1][0]:
            raise ValueError(
                "Arguments of Config should be sorted in increasing alphabetic "
                f"order. Found '{args[i][0]}' before '{args[i + 1][0]}'. "
                f"Config: {config}"
            )
