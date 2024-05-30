# API Pack A Massive Multi-Programming Language Dataset for API Call Generation

This is the repository with dataset, code, and models for the paper [API Pack A Massive Multi-Programming Language Dataset for API Call Generation](https://github.com/zguo0525/API-Pack)

## Dataset Summary

API Pack is a large-scale, multi-programming language dataset containing over 1 million instances across 10 programming languages for API call generation and intent detection. Its key features include multilinguality, scale, and a wide range of real-world APIs and use cases, enabling the assessment of cross-lingual skill transfer. Evaluation experiments demonstrated that CodeLlama-13B, fine-tuned with only 20,000 Python instances from API Pack, outperformed GPT-3.5 and GPT-4 in generating API calls for entirely new APIs, highlighting the dataset's effectiveness in improving the API call generation capabilities of large language models.

## Dataset Access

The dataset in JSON format is hosted on [Huggingface](https://huggingface.co/datasets/zguo0525/API-Pack).

## Dataset Structure

## License

API Pack dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License. You are free to share and adapt the material under the terms that you must give appropriate credit, provide a link to the license, and indicate if changes were made.
The code under this repo is licensed under an MIT License.

## Disclaimer

This dataset was collected and released solely for research purposes to improve open-source large language models' API call generation capabilities. The authors are strongly against any potentially harmful use of the data or technology by any party.

## Citation

If you find our dataset useful, please consider citing the paper:

```
@misc{guo2024api,
      title={API Pack: A Massive Multilingual Dataset for API Call Generation}, 
      author={Zhen Guo and Adriana Meza Soria and Wei Sun and Yikang Shen and Rameswar Panda},
      year={2024},
      eprint={2402.09615},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
