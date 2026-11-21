def coerce_types(config):
    """
    Recursively scans the config and converts strings like '1e-4' 
    or '32' into floats/ints automatically.
    """
    if isinstance(config, dict):
        for k, v in config.items():
            config[k] = coerce_types(v)
    elif isinstance(config, list):
        return [coerce_types(x) for x in config]
    elif isinstance(config, str):
        # 1. Check if it's a digit (int)
        if config.isdigit():
            return int(config)
        # 2. Check if it's a float or scientific notation
        try:
            # float() natively handles '1e-6'
            return float(config)
        except ValueError:
            # If it's a real string (e.g., 'SpanGuidedDPI'), keep it as is
            return config
    return config