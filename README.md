# Audio Replay Attacks Repository

This repository focuses on detecting replay attacks in audio. The models provided are trained to receive audio files as input and classify whether the audio is legitimate or a replay attack.

## Repository Structure

The repository is organized into two main folders:
  
- **`notebooks/`**: Contains notebooks for quick code execution and testing, such as data download, data processing, model training, model loading, and saving.
  
- **`scripts/`**: Includes the scripts required to execute the complete process, including data download, data processing, model training, and model saving, so that these tasks can be run directly from a terminal (on a local PC, server, etc.).
  - **`scripts/advanced_models/`**: Contains the training pipelines and experiments for the most performant models in the project, particularly the Wav2Vec2-based approach. This includes different training configurations: a baseline version without data augmentation, a full data augmentation setup to maximize robustness, and a reduced augmentation variant designed to lower computational cost while maintaining competitive performance.
  - **`scripts/data_processing/`**: Contains all scripts related to dataset preparation and transformation. This includes dataset cleaning and normalization, generation of telephone-band versions of the audio data (e.g., resampling, filtering, and codec simulation), and the creation and management of dataset annotations and labels required for training and evaluation.

## Notebooks

Use the notebooks for:

- Dataset downloads that require manual steps or authentication.
- Exploratory data analysis and pre-processing experiments.
- Prototyping advanced model training before converting to scripts.

## Advanced models

Scripts and experiments for higher-capacity models (deep learning, Vision Transformers for spectrogram input, etc.) are in `scripts/advanced_models/`. These experiments typically require more data, GPU resources and additional configuration. Check the notebooks in `notebooks/` for example training runs and configuration hints.
