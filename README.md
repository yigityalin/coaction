# coaction
A library for multi-agent learning that aims to accelerate the research.


## What is coaction for?
coaction can create and run simulations for any Markov game whose stage ransitions and rewards can be represented by multi-dimensional array. It provides implementations of popular learning dynamics such as fictitious play and individual Q-learning. Since the library is created for research, it also allows users to implement their own agents and run the simulations without the extra burden of parallelization and logging.


## How to use
Though coaction can be used similar to other Python libraries, coaction provides two main functionalities: creation of the project configurations and running the simulations for given configurations.

To create the project directory and the configuration files, one can use coaction.create. An example script is given below.

```
python -m coaction.create \
    --parent_dir ../ \
    --project_name example_project \
    --experiment_name example_experiment_1 example_experiment_2 example_experiment_3 example_experiment_4 \
    --agent_types SynchronousFictitiousPlay SynchronousFictitiousPlay \
    --agent_types CustomAgent SynchronousFictitiousPlay \
    --agent_types AsynchronousFictitiousPlay AsynchronousFictitiousPlay \
    --agent_types AsynchronousFictitiousPlay AnotherCustomAgent \
    --game_types MatrixGame MatrixGame MarkovGame MarkovGame
```

***parent_dir*** is the directory in which the project will be created.

***project_name*** is the name of the project.

***experiment_name*** is a list of experiment names.

***agent_types*** is the classes of agents that will be used in the experiments. Note that each entry of agent types is for the corresponding experiment you listed via the ***experiment_name*** argument. For custom agents, write the name of the custom agent class you will implement. coaction will create a template for your custom agent.

***game_types*** is either "MatrixGame" or "MarkovGame." Recall that Markov games contains multiple matrix games as its stage games. Internally, all the games are converted to a Markov game. This distinction removes the burden of creating a transition matrix of all ones for matrix games.

After the creation of the configuration files, you will find templates that you need to complete before running the experiment. When you complete the configuration files, you can run the project as given below:

```
python -m coaction.run --project ../example_project
```

This script will start the experiments according to your configuration. coaction also copies the configuration files to the corresponding log directory. This allows you to change the configuration of the project without losing the information about your previous runs.
