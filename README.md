Install pre-commit and run `pre-commit run --all-files`


### PDF Download scripts :
There are two ways to download the files -

1. URL : Document href from the document metadata
2. `https://api.sustainabilityreportingnavigator.com/api/documents/{doc_id}/download` : Provide the document  id in the url

Script folder : `srn_scrape`

### PDF parsing script
This script will convert the downloaded pdfs into json format using the viper parser.

Install viper paser from [here](https://gitlab.cc-asp.fraunhofer.de/ppradhan/viper.git)

checkout branch - `viper-optimizations`

1. Set the file paths in viper_config.yaml
2. Run the python script : `python viper_parser.py --config /cluster/home/repo/my_llm_experiments/esrs_data_scraping/viper_config.yaml --cuda-ids 0 1 2 3 4 5 6 --num_workers 8`


### Database creation script:

1. set the path locations in main.yaml
2. Run the python script : `python main.py --config /cluster/home/repo/my_llm_experiments/esrs_data_scraping/main.yaml`

Key notes:
1. All functions with prefix `collect_` in `main.py` downloads the data from the api and pushes the data to the sqlite db.
2. The model_name = "gpt-3.5-turbo" - is a paid api for first run `convert_pdf_est_json` function in main.py to find out the cost.
3. `json_to_docx` function in `main.py` converts the parsed json file into docx documents with a unique identifier  
