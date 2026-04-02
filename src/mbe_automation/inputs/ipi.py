from mbe_automation.configs.dynamics import DynamicsConfig


def dynamics(config: DynamicsConfig, template_dir: str = None) -> str:
    """
    Generate i-PI input XML string for molecular dynamics.
    
    Args:
        config: DynamicsConfig object containing all simulation parameters
    
    Returns:
        XML string for i-PI input
    """

    is_quantum = config.n_beads > 1
    ensemble_type = config.ensemble.lower()
    template_name = f"{ensemble_type}_{'quantum' if is_quantum else 'classical'}.xml"
    template_path = template_name    
    with open(template_path, 'r') as f:
        template = f.read()

    params = {
        **config.__dict__,
        'sampling_stride': int(config.sampling_interval_fs / config.time_step_fs),
        'total_steps': int(config.time_total_fs / config.time_step_fs),
        'equilibration_steps': int(config.time_equilibration_fs / config.time_step_fs),
        'ensemble_lower': config.ensemble.lower(),
        'pressure_bar': config.pressure_GPa * 10000,
    }
    
    # Handle PILE thermostat lambda parameter
    if config.thermostat in ["pile_l", "pile_g"]:
        params['pile_lambda_block'] = '<pile_lambda>0.5</pile_lambda>'
    else:
        params['pile_lambda_block'] = ''
    
    if config.thermostat in ["pile_l", "pile_g"]:
        params['thermostat_extras'] = '<pile_lambda>0.5</pile_lambda>'
    elif config.thermostat == "gle":
        params['thermostat_extras'] = '<A_matrix>GLE_A_matrix.txt</A_matrix>\n          <C_matrix>GLE_C_matrix.txt</C_matrix>'
    else:
        params['thermostat_extras'] = ''
    
    return template.format(**params)
