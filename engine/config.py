"""TOML experiment presets using only argparse and tomllib."""
import argparse
from pathlib import Path
import tomllib

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / 'configs/experiments.toml'
ARCHITECTURES = ('residual', 'pooled', 'attention')
FIELDS = {'board_size', 'connect', 'stones_per_turn', 'starter_stones', 'channels', 'blocks',
          'workers', 'parallel', 'simulations', 'batch', 'seed', 'learning_rate', 'replay_limit',
          'bootstrap_games', 'updates_per_cycle', 'tactical_ms', 'snapshot_every', 'hours',
          'disk_gib', 'device', 'seconds', 'data', 'max_games', 'architecture'}
INTEGER_FIELDS = {'connect', 'stones_per_turn', 'starter_stones', 'channels', 'blocks', 'workers',
                  'parallel', 'simulations', 'batch', 'seed', 'replay_limit', 'bootstrap_games',
                  'updates_per_cycle', 'snapshot_every', 'max_games'}
TEXT_FIELDS = {'board_size', 'device', 'data', 'architecture'}


def load_options(argv):
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    early.add_argument('--preset', default='gomoku')
    selected, _ = early.parse_known_args(argv)
    with selected.config.open('rb') as stream:
        config = tomllib.load(stream)
    if set(config) - {'defaults', 'presets'} or not isinstance(config.get('defaults'), dict) or not isinstance(config.get('presets'), dict):
        raise ValueError('Config needs [defaults] and [presets.NAME] tables only.')
    if selected.preset not in config['presets']:
        raise ValueError(f'Unknown preset: {selected.preset}. Choose: {", ".join(config["presets"])}')
    for values in [config['defaults'], *config['presets'].values()]:
        if not isinstance(values, dict) or set(values) - FIELDS:
            raise ValueError('Unknown setting or invalid preset table in TOML config.')
        for key, value in values.items():
            valid = type(value) is int if key in INTEGER_FIELDS else isinstance(value, str) if key in TEXT_FIELDS else type(value) in (int, float)
            if not valid:
                raise ValueError(f'Invalid config type for {key}.')
    options = {**config['defaults'], **config['presets'][selected.preset]}
    explicit = {item.split('=')[0].lstrip('-').replace('-', '_') for item in argv if item.startswith('--')}
    if 'config' in explicit or 'preset' in explicit:
        explicit.update(options)
    return selected, options, explicit
