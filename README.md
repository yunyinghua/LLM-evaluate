# LLM-evaluate

A lightweight yet powerful Python script for benchmarking large language models.  
Just plug in your API key and base URL, install the dependencies, and run evaluations against any dataset placed in the `dataset/` folder.

## Start

1. Install requirements  
   ```bash
   pip install -r requirements.txt
   ```

2. Configure credentials  
   ```python
   API_KEY  = "YOUR_API_KEY_HERE"
   BASE_URL = "https://api.your-provider.com/v1"
   ```

3. Download datasets  
   download from https://huggingface.co/datasets/MMMU/MMMU/tree/main  
   Remember set the path of your datasets  
   ```python
   data_files="Computer_Science/validation-00000-of-00001.parquet"
   ```
4. Launch the evaluation  
   ```bash
   python mmmu_evaluate.py
   ```

5. Results will be saved to "evaluation_results.csv"
