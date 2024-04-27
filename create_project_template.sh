python -m coaction.create \
    --parent_dir ../ \
    --project_name example_project \
    --experiment_name example_experiment_1 example_experiment_2 example_experiment_3 example_experiment_4 \
    --agent_types SynchronousSmoothedFictitiousPlay SynchronousFictitiousPlay \
    --agent_types SynchronousFictitiousPlay CustomAgent \
    --agent_types AsynchronousSmoothedFictitiousPlay ModelFreeSmoothedFictitiousPlay \
    --agent_types ModelFreeFictitiousPlay CustomAgent2 \
    --game_types MatrixGame MatrixGame MarkovGame MarkovGame