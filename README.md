# Fine-tuning StarCoder2 for Java Code Documentation and Optimization

This notebooks demonstrate how to fine-tune the StarCoder2-3b & unsloth/Qwen2.5-Coder-14B-Instruct-128K-GGUF model and use Terjman as well as codellama for generating documentation and optimizing Java code using the `instructional_code-search-net-java` dataset and the Hugging Face Transformers library, given instructions in either Darija or English. The fine-tuned model is then deployed to Baseten Cloud using Truss.

## Requirements

* Google Colab environment
* Hugging Face account and API token (stored in Colab userdata as 'HF_TOKEN')
* Python libraries: `datasets`, `accelerate`, `evaluate`, `torch`, `transformers`, `timm`, `rouge_score`, `jiwer`, `peft`, `bitsandbytes`, `trl`, `truss`
* Baseten account

## Process

1. **Installation:** Install the necessary libraries using pip.
2. **Hugging Face Access:** Obtain your Hugging Face API token and store it in Colab userdata.
3. **Data Preparation:** Load the `instructional_code-search-net-java` dataset and preprocess it for fine-tuning.
4. **Model and Tokenizer:** Load the StarCoder2-3b model and tokenizer, applying 4-bit quantization for efficiency.
5. **Training Arguments:** Set up training parameters like sequence length, batch size, learning rate, etc.
6. **Metrics:** Define evaluation metrics including accuracy, perplexity, ROUGE, BLEU, and others.
7. **Trainer:** Utilize the `SFTTrainer` from the TRL library for efficient fine-tuning with LoRA.
8. **Fine-tuning:** Initiate the training process and monitor progress.
9. **Evaluation:** Evaluate the fine-tuned model on a held-out dataset and print the results.
10. **Model Deployment:** Package the model using Truss and deploy to Baseten Cloud.
11. **Inference:** Use the deployed model to generate documentation and optimize Java code through the Baseten platform.


## Usage

1. Open the notebook in Google Colab.
2. Ensure your Hugging Face API token is stored in Colab userdata under the key 'HF_TOKEN'.
3. Execute the code cells sequentially.
4. In the final code cell, provide your Java code snippet as input to the `generate_documentation_and_optimization` function to obtain the model's response locally.
5. For using the deployed model, refer to the Baseten documentation for accessing and interacting with the deployed Truss package.

## Notes

* The model is fine-tuned using LoRA for efficiency.
* The training process can take a significant amount of time depending on the resources available.
* The output quality may vary depending on the complexity and style of the input Java code.
* Refer to Truss documentation for detailed instructions on packaging and deploying the model to Baseten.


## Models Used

* **BSJCode-1-Stable:** `BSAtlas/BSJCode-1-Stable`
* **CodeLlama:** `codellama/CodeLlama-7b-Instruct-hf`
* **Terjman:** `atlasia/Terjman-Ultra`


## Deployment

The fine-tuned model is deployed to Baseten Cloud using Truss. Truss is a framework for building and deploying machine learning models as API services. Baseten provides a platform for hosting and managing those services.

To deploy the model, follow these steps:

1. Create a Truss package for the model.
2. Configure the package for Baseten Cloud.
3. Deploy the package to Baseten Cloud.

Refer to the Truss and Baseten documentation for detailed instructions on deployment.


## Acknowledgments

* Hugging Face Transformers library
* TRL library for efficient fine-tuning
* The authors of the `instructional_code-search-net-java` dataset
* StarCoder2-3b model developers
* Truss framework
* Baseten Cloud platform
