import os.path as op

configfile: "config/snakemake.yml"

ruleorder: bcm_indexes > merge_bcm > convert_bcm > \
    projection_indexes > aggregate_projection > \
    merge_projection > convert_projection > to_netcdf

# Directories
projdir = op.join(config['root_dir'], config['projections_subdir'])
mortdir = op.join(config['root_dir'], config['mortality_subdir'])
bcmdir = op.join(config['root_dir'], config['bcm_subdir'])
resultsdir = op.join(config['root_dir'], config['results_subdir'])
figuresdir = op.join(config['root_dir'], config['figures_subdir'])
topodir = op.join(config['root_dir'], config['topo_subdir'])

# Climate Files
monthly_dataset = op.join(bcmdir, 'BCMv8_monthly.zarr')
annual_dataset = op.join(bcmdir, 'BCMv8_annual.zarr')
index_dataset = op.join(bcmdir, 'BCMv8_indexes.zarr')

# Mortality Files
def mortfile(base): return op.join(mortdir, 'generated', base)

mort = mortfile('tree_mortality.zarr')
mort_folds = mortfile('tree_mortality_folds.zarr')
mort_training = mortfile('tree_mortality_training.zarr')
mort_training_nonzero = mortfile('tree_mortality_training_nonzero.zarr')
mort_rand_folds = mortfile('tree_mortality_random_folds.zarr')
mort_rand_training = mortfile('tree_mortality_random_training.zarr')
mort_rand_training_nonzero = mortfile('tree_mortality_random_training_nonzero.zarr')

mortfiles = [
    mort, mort_folds, mort_training, mort_training_nonzero,
    mort_rand_folds, mort_rand_training, mort_rand_training_nonzero
]

# Topography Files
topo_gen = op.join(topodir, 'generated')
topofile = op.join(topodir, 'topo_indexes.zarr')
demfile = op.join(topodir, config['topobase'])

# Result Files
tm_results = op.join(resultsdir, 'tree_mortality_loo.npz')
tm_random_results = op.join(resultsdir, 'tree_mortality_random_loo.npz')


rule all_training:
    input:
        tm_results


rule train_model:
    input:
        op.join(mortdir, 'generated', '{base}_training.zarr')
    output:
        op.join(resultsdir, '{base}_loo.npz')
    shell:
        "python src/train_rf_model_ray.py {input} {output}"


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
        index_dataset,
        topofile
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


rule topo:
    input:
        features=topo_gen,
        bcm=annual_dataset
    output:
        directory(topofile)
    params:
        conf=config['topo_config']
    shell:
        "python src/merge_topo_features.py {input.features}/*.nc {input.bcm} {params.conf} {output}"


rule topo_features:
    input:
        demfile
    output:
        directory(topo_gen)
    shell:
        """
        mkdir -p {output}
        Rscript r/raster_topography_indexes.r -i {input} -o {output}
        """


rule all_projections:
    input:
        expand(
            op.join(projdir, '{model}', '{scenario}{suffix}.nc4'),
            model=config['projection_models'],
            scenario=config['projection_scenarios'],
            suffix=['', '_annual', '_indexes'],
        )


rule projection_indexes:
    input:
        op.join(projdir, '{model}', '{scenario}_annual.zarr')
    output:
        directory(op.join(projdir, '{model}', '{scenario}_indexes.zarr'))
    params:
        conf=config['bcm_proj_ind_config'],
        ref=annual_dataset
    shell:
        "python src/append_climate_indexes.py {input} {params.conf} {output} -r {params.ref}"


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


rule all_bcm:
    input:
        expand(
            op.join(bcmdir, 'BCMv8_{suffix}.nc4'),
            suffix=['monthly', 'annual', 'indexes'],
        )


rule bcm_indexes:
    input:
        annual_dataset
    output:
        directory(index_dataset)
    params:
        config['bcm_ind_config']
    shell:
        "python src/append_climate_indexes.py {input} {params} {output}"


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
    params:
        config['bcm_config']
    shell:
        "python src/merge_bcm.py {input} {params} {output}"


use rule merge_bcm as merge_projection with:
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


rule to_netcdf:
    priority: -1
    input:
        '{base}.zarr'
    output:
        '{base}.nc4'
    shell:
        "python src/zarr_to_nc4.py {input} {output}"
