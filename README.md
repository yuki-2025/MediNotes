# Research Capstone Project: MediNotes
MediNotes: SOAP Note Generation through Ambient Listening, Large Language Model Fine-Tuning, and RAG
- MediNotes was awarded “Best in Show” as one of the Top Capstone Projects at the University of Chicago showcase.
- This project was a collaboration with UChicago Medicine to advance healthcare AI.
- Building on groundbreaking research from the Microsoft AI team published in Nature, we developed an innovative framework designed to streamline medical documentation and the consultation process, with the goal of alleviating physician burnout.
- By combining cutting-edge technologies like ambient listening, large language model fine-tuning, and retrieval-augmented generation (RAG), MediNotes represents a significant step forward in optimizing healthcare workflows and improving physician efficiency.

## Demo Preview: 
<a href="http://www.youtube.com/watch?feature=player_embedded&v=QmWfvFdQc08
" target="_blank"><img src="http://img.youtube.com/vi/QmWfvFdQc08/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="640" height="360" border="10" /></a>

## Live Demo
This may contain error for preview for raw demo because you may need GPU more to 4gb to run the model: <br>
https://medinotes-llm.streamlit.app/search

## Code Setup 
1. Download the requirements.txt and app.py file.
2. Install the necessary libraries.

```python
!pip install -r requirements.txt  
```

3. Run the app.

```python 
python -m streamlit run app.py
```
 
## Model:
You can try the fine-tuned model yourself, which converts medical dialogues into SOAP notes:

Full 4-bit quantized model:
https://huggingface.co/Yuki20/llama3_8b_aci_3e_full4b

Full model:
https://huggingface.co/Yuki20/llama3_8b_aci_3e_full

Adapter only with Unsloth:  <br>
https://huggingface.co/Yuki20/mistral_7b_aci_3e  <br>
https://huggingface.co/Yuki20/llama3_8b_aci_5e  <br>
https://huggingface.co/Yuki20/llama3_8b_aci  <br>

## Citation
If you find Medinotes useful in your research or applications, please kindly cite:
```

@inproceedings{leong2024medinotes, 
title={{MediNotes}: A Generative AI Framework for Medical Note Generation}, 
author={Leong, HY and Gao, YF and Ji, S and Kalaycioglu, Bora and Pamuksuz, Uktu}, 
journal={arXiv preprint arXiv:2410.01841}, 
year={2024} }
```

## Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository, Aci-bench. Thanks for their wonderful works.
