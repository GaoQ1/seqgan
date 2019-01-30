from configparser import SafeConfigParser

def get_config(config_file='config.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)

    # get the ints, floats, strings and lists
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    _conf_lists = [(key, [int(item) for item in value.split(',')]) for key, value in parser.items('lists')]

    return dict(_conf_ints + _conf_floats + _conf_strings + _conf_lists)
