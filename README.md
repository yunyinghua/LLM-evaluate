# LLM-evaluate

A lightweight yet powerful Python script for benchmarking large language models.  
Just plug in your API key and base URL, install the dependencies, and run evaluations against any dataset placed in the `dataset/` folder.

## Quick Start

1. Install requirements  
   ```bash
   pip install -r requirements.txt
   ```

2. Configure credentials  
   ```python
   API_KEY  = "YOUR_API_KEY_HERE"
   BASE_URL = "https://api.your-provider.com/v1"
   ```

4. Launch the evaluation  
   ```bash
   python mmmu_evaluate.py
   ```

5. Results will be saved to "evaluation_results.csv"
