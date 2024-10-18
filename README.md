0. Python 3.7 is required to run this project.
    (Optional: use a virtual environment)

1. Place the required files in the `Data` folder:
  - A TSV file with entity pairs in the format (Ent1, Ent2, Label)
  - The annotations file
  - The ontology files in both OBO and OWL formats

2. Set the paths and configs in `constants.py`

3. Install the required packages:
  - Run `install_requirements.py`

4. Generate KG embeddings and semantic similarities:
  - Run `Generate_KGE_and_SSM/generate_kge_and_ssm.py`

5. Generate statistics:
  - Run `Statistics/run_stats.py`

6. Train models on the generated embeddings and semantic similarities:
  - Run `Train_Models/run_training.py`