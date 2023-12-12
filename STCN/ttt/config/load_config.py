import argparse
import yaml
import os

"""
Inspired from https://gist.github.com/robcowie/7c39e2afb140c905fdf2661b93ff3df9
"""


class ConfigAction(argparse.Action):
    """Add configuration file to current defaults."""

    def __init__(self, *args, **kwargs):
        """Config action is a search path, so a list, so one or more argument."""
        super().__init__(*args, nargs='+', **kwargs)

    def __call__(self, parser, ns, values, option):
        """Change defaults for the namespace.
        still allows overriding from commandline options
        """
        for path in values:
            parser.set_defaults(**self.parse_config(path))

    def parse_config(self, path):
        """Abstract implementation of config file parsing."""
        raise NotImplementedError()


class YamlConfigAction(ConfigAction):
    """YAML config file parser action."""

    def parse_config(self, path):
        try:
            with open(os.path.expanduser(path), 'r') as handle:
                return self.reformat(yaml.safe_load(handle))
        except (FileNotFoundError, yaml.parser.ParserError) as e:
            raise argparse.ArgumentError(self, e)

    def reformat(self, yaml_dict):
        """
        flattens and lowercase
        """
        reformatted = {}
        for section_key in yaml_dict:
            for key in yaml_dict[section_key]:
                reformatted[key.lower()] = yaml_dict[section_key][key]
        return reformatted


class ConfigArgumentParser(argparse.ArgumentParser):
    """Argument parser which supports parsing extra config files.
    Config files specified on the commandline through the
    YamlConfigAction arguments modify the default values on the
    spot. If a default is specified when adding an argument, it also
    gets immediately loaded.
    This will typically be used in a subclass, like this:
            self.add_argument('--config', action=YamlConfigAction, default=self.default_config())
    """

    def _add_action(self, action):
        # this overrides the add_argument() routine, which is where
        # actions get registered. it is done so we can properly load
        # the default config file before the action actually gets
        # fired. Ideally, we'd load the default config only if the
        # action *never* gets fired (but still setting defaults for
        # the namespace) but argparse doesn't give us that opportunity
        # (and even if it would, it wouldn't retroactively change the
        # Namespace object in parse_args() so it wouldn't work).
        action = super()._add_action(action)
        if isinstance(action, ConfigAction) and action.default is not None:
            # fire the action, later calls can override defaults
            try:
                action(self, None, action.default, None)
            except argparse.ArgumentError:
                # ignore errors from missing default
                pass

def load_config():
    parser = ConfigArgumentParser('test')
    parser.add_argument('--config_name', default='stcn_s01_mcc')
    # DATASET PARAMETERS
    parser.add_argument('--dataset_name', default='davis', choices=['davis', 'youtube', 'mose', 'davis-c'])
    parser.add_argument('--dataset_dir', default='prout', help='Path to the dataset')
    parser.add_argument('--corrupted_image_dir', default=None, help='Path to DAVIS-C corruption images')
    parser.add_argument('--split', default='val', choices=['val', 'testdev', 'valid', 'train', 'test'],
                        help='Split of the dataset to be processed')
    parser.add_argument('--palette_video_name', default=None, type=str,
                        help='Mask to get the palette colours')
    parser.add_argument('--video_set_filename', default=None, type=str,
                        help='Filename containing a subset of videos to process')
    parser.add_argument('--output_dir', help='Path to output folder')
    parser.add_argument('--seed', default=1, type=int, help='Seed used for the random number generators')
    namespace, _ = parser.parse_known_args()

    config_filename = os.path.join("ttt_configs", f'{namespace.config_name}.yaml')
    parser.add_argument('--config', action=YamlConfigAction, default=[config_filename])
    return parser.parse_args()


if __name__ == "__main__":
    args = load_config()
    for arg in vars(args):
        print(f"arg {arg}, {getattr(args, arg)}")
    print(args.dataset_name)
