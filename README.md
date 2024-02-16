# API Pack

Welcome to API Pack, a comprehensive multilingual dataset designed to significantly advance the API call generation capabilities of large language models. With over one million instruction-API call pairs, API Pack serves as a robust resource for researchers and developers aiming to enhance large language models' proficiency in both specific API call generation tasks and general coding abilities.

## Overview

API Pack is rooted in the goal of improving the interaction between large language models and various programming interfaces. By providing a vast collection of instruction-API call pairs, API Pack facilitates a more nuanced understanding and generation of API calls, a critical aspect of programming and software development.

## Key Features

- **Extensive Dataset:** Over one million instruction-API call pairs, catering to a wide range of API call generation scenarios.
- **Multilingual Support:** Designed to advance cross-lingual API call generation capabilities without the need for extensive language-specific data.
- **Proven Efficacy:** Experiments demonstrate that fine-tuning models with API Pack significantly enhances their performance in generating unseen API calls. Fine-tuning CodeLlama-13B on just 20,000 Python instances results in over 10% and 5% higher accuracy compared to GPT-3.5 and GPT-4, respectively.
- **Improved Generalization:** Scaling the training process to 100,000 examples further boosts the models' ability to generalize to new APIs not encountered during training.

## Getting Started

To get started with API Pack, clone this repository using the following command:

```bash
git clone https://github.com/{anonymous url}
```

Please replace `{anonymous url}` with the actual URL of this repository.

### Installation

After cloning the repository, navigate to the API Pack directory and install the required dependencies:

```bash
cd API-Pack
pip install -r requirements.txt
```

### Usage

Refer to the documentation and example scripts provided in the repository to understand how to utilize the dataset and fine-tuned models for your API call generation tasks.

## Experiments and Results

Our extensive experiments underline the effectiveness of API Pack in enhancing large language models' performance in API call generation. Detailed results and analysis are available in the `experiments` directory.

## Contribution

Contributions to API Pack are welcome! Whether it's extending the dataset, improving the models, or refining the documentation, your input is valuable. Please refer to the `CONTRIBUTING.md` file for more details on how to contribute.

## License

This project is released under the MIT License. See the `LICENSE` file for more details.

## Contact

For any inquiries or further discussion related to API Pack, please open an issue on this repository.
