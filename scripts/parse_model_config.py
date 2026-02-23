#!/usr/bin/env python3
"""
Parse model configuration to determine which files are required for inference.

Reads model configuration from log.md or config file to determine:
- Whether prediction mode is enabled (no labels required)
- Whether pretrained models are used (requires antigen_sequence.json)
- Whether AntiBERTy is used (requires cdr_sequence.json)
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional


def parse_log_md(log_path: str) -> Dict[str, Any]:
    """Parse log.md file to extract configuration."""
    config = {
        'prediction': False,
        'use_pretrained': False,
        'use_antiberty': False,
        'pretrained_model': None,
        'max_antigen_len': None,
    }
    
    if not os.path.exists(log_path):
        return config
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Parse key-value pairs
        # Look for patterns like "use_pretrained: True" or "use_pretrained = True"
        patterns = {
            'prediction': r'prediction[:\s=]+(True|False|true|false)',
            'use_pretrained': r'use_pretrained[:\s=]+(True|False|true|false)',
            'use_antiberty': r'use_antiberty[:\s=]+(True|False|true|false)',
            'pretrained_model': r'pretrained_model[:\s=]+([^\s\n]+)',
            'max_antigen_len': r'max_antigen_len[:\s=]+(\d+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1)
                if key in ['prediction', 'use_pretrained', 'use_antiberty']:
                    config[key] = value.lower() in ['true', '1', 'yes']
                elif key == 'max_antigen_len':
                    config[key] = int(value)
                else:
                    config[key] = value
        
    except Exception as e:
        print(f"Warning: Could not parse log.md: {e}", file=sys.stderr)
    
    return config


def parse_config_json(config_path: str) -> Dict[str, Any]:
    """Parse JSON configuration file."""
    config = {
        'prediction': False,
        'use_pretrained': False,
        'use_antiberty': False,
        'pretrained_model': None,
        'max_antigen_len': None,
    }
    
    if not os.path.exists(config_path):
        return config
    
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Extract relevant fields
        for key in config.keys():
            if key in data:
                config[key] = data[key]
    
    except Exception as e:
        print(f"Warning: Could not parse config JSON: {e}", file=sys.stderr)
    
    return config


def get_model_config(model_dir: str) -> Dict[str, Any]:
    """
    Get model configuration from model directory.
    
    Looks for:
    1. log.md file in model directory
    2. config.json file in model directory
    3. Default values if neither found
    """
    config = {
        'prediction': False,
        'use_pretrained': False,
        'use_antiberty': False,
        'pretrained_model': None,
    }
    
    if not os.path.exists(model_dir):
        return config
    
    # Try log.md first
    log_path = os.path.join(model_dir, 'log.md')
    if os.path.exists(log_path):
        parsed = parse_log_md(log_path)
        config.update(parsed)
    
    # Try config.json
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        parsed = parse_config_json(config_path)
        config.update(parsed)
    
    return config


def get_required_files(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Determine which files are required based on configuration.
    
    Returns:
        Dictionary mapping file names to required status
    """
    required = {
        'node_feature.parquet': True,
        'edge_index.parquet': True,
        'edge_attribute_dist.parquet': True,
        'edge_attribute_charge.parquet': True,
        'node_label_pi.parquet': not config.get('prediction', False),
        'antigen_sequence.json': config.get('use_pretrained', False),
        'cdr_sequence.json': config.get('use_antiberty', False),
    }
    
    return required


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse model configuration")
    parser.add_argument('--model_dir', type=str, help='Path to model directory')
    parser.add_argument('--log_file', type=str, help='Path to log.md file')
    parser.add_argument('--config_file', type=str, help='Path to config.json file')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    config = {}
    
    if args.model_dir:
        config = get_model_config(args.model_dir)
    elif args.log_file:
        config = parse_log_md(args.log_file)
    elif args.config_file:
        config = parse_config_json(args.config_file)
    else:
        print("Error: Must provide --model_dir, --log_file, or --config_file")
        return 1
    
    # Get required files
    required_files = get_required_files(config)
    
    # Output results
    output_data = {
        'config': config,
        'required_files': required_files
    }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
    else:
        print(json.dumps(output_data, indent=2))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

