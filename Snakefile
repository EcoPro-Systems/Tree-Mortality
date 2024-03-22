import os.path as op

configfile: "config/snakemake.yml"

ruleorder: aggregate_projection > merge_projection > convert_projection > to_netcdf

# Directories
projdir = op.join(config['root_dir'], config['projections_subdir'])
mortdir = op.join(config['root_dir'], config['mortality_subdir'])
bcmdir = op.join(config['root_dir'], config['bcm_subdir'])

# Climate Files
monthly_dataset = op.join(bcmdir, 'BCMv8_monthly.zarr')
annual_dataset = op.join(bcmdir, 'BCMv8_annual.zarr')
index_dataset = op.join(bcmdir, 'BCMv8_indices.zarr')

# Mortality Files
mort = op.join(
    mortdir, 'generated', 'tree_mortality.zarr'
)
mort_folds = op.join(
    mortdir, 'generated', 'tree_mortality_folds.zarr'
)
mort_training = op.join(
    mortdir, 'generated', 'tree_mortality_training.zarr'
)
mort_training_nonzero = op.join(
    mortdir, 'generated', 'tree_mortality_training_nonzero.zarr'
)

mort_rand_folds = op.join(
    mortdir, 'generated', 'tree_mortality_random_folds.zarr'
)
mort_rand_training = op.join(
    mortdir, 'generated', 'tree_mortality_random_training.zarr'
)
mort_rand_training_nonzero = op.join(
    mortdir, 'generated', 'tree_mortality_random_training_nonzero.zarr'
)

mortfiles = [
    mort, mort_folds, mort_training, mort_training_nonzero,
    mort_rand_folds, mort_rand_training, mort_rand_training_nonzero
]


rule all_mortality:
    input:
        expand(
            '{base}.nc4',
            base=glob_wildcards('{base}.zarr', files=mortfiles).base
        )


rule nonzero_mortality:
    input:
        mort_training
    output:
        directory(mort_training_nonzero)
    params:
        config['mort_trainset_config']
    shell:
        "python src/filter_zero_values.py {input} {params} {output}"


use rule nonzero_mortality as nonzero_random_mortality with:
    input:
        mort_rand_training
    output:
        directory(mort_rand_training_nonzero)


rule mortality_training:
    input:
        mort_folds,
        index_dataset
    output:
        directory(mort_training)
    params:
        config['mort_trainset_config']
    shell:
        "python src/construct_training_dataset.py {input} {params} {output}"


use rule mortality_training as mortality_random_training with:
    input:
        mort_rand_folds,
        index_dataset
    output:
        directory(mort_rand_training)


rule mortality_folds:
    input:
        mort
    output:
        directory(mort_folds)
    params:
        config['mort_fold_config']
    shell:
        "python src/append_folds.py {input} {params} {output}"


use rule mortality_folds as mortality_random_folds with:
    output:
        directory(mort_rand_folds)
    params:
        config['mort_rand_fold_config']


rule mortality:
    input:
        mortdir
    output:
        directory(mort)
    params:
        config['mort_config']
    shell:
        "python src/convert_tree_mortality.py {input} {params} {output}"


rule all_projections:
    input:
        expand(
            op.join(projdir, '{model}', '{scenario}{suffix}.nc4'),
            model=config['projection_models'],
            scenario=config['projection_scenarios'],
            suffix=['', '_annual'],
        )


rule aggregate_projection:
    input:
        op.join(projdir, '{model}', '{scenario}.zarr')
    output:
        directory(op.join(projdir, '{model}', '{scenario}_annual.zarr'))
    params:
        config['agg_config']
    shell:
        "python src/aggregate_bcm_v8.py {input} {params} {output}"


use rule aggregate_projection as aggregate_bcm with:
    input:
        monthly_dataset
    output:
        directory(annual_dataset)


rule bcm_indexes:
    input:
        annual_dataset
    output:
        directory(index_dataset)
    params:
        config['bcm_ind_config']
    shell:
        "python src/append_climate_indices.py {input} {params} {output}"


rule merge_projection:
    input:
        expand(
            op.join(
                projdir, '{{model}}', '{{scenario}}',
                '{var}_{{model}}_{{scenario}}.nc4'
            ),
            var=config['bcm_variables'],
        )
    output:
        directory(op.join(projdir, '{model}', '{scenario}.zarr'))
    shell:
        "python src/merge_projections.py {input} {output}"


rule convert_projection:
    input:
        op.join(
            projdir, '{model}', '{scenario}',
            '{var}_{model}_{scenario}.zip'
        )
    output:
        op.join(
            projdir, '{model}', '{scenario}',
            '{var}_{model}_{scenario}.nc4'
        )
    params:
        config['bcm_config']
    shell:
        "python src/convert_projections.py {input} {params} {output}"


rule merge_bcm:
    input:
        expand(
            op.join(
                bcmdir, config['bcm_raw_subdir'],
                '{var}.nc4'
            ),
            var=config['bcm_variables']
        )
    output:
        directory(monthly_dataset)
    shell:
        "python src/merge_projections.py {input} {output}"


rule convert_bcm:
    input:
        op.join(bcmdir, config['bcm_raw_subdir'], '{var}')
    output:
        op.join(
            bcmdir, config['bcm_raw_subdir'],
            '{var}.nc4'
        )
    params:
        config['bcm_config']
    shell:
        "python src/convert_bcm_v8.py {input} {params} {output}"


rule to_netcdf:
    input:
        '{base}.zarr'
    output:
        '{base}.nc4'
    shell:
        "python src/zarr_to_nc4.py {input} {output}"
