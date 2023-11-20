Install pre-commit and run `pre-commit run --all-files`


### PDF Download scripts :
There are two ways to download the files -

1. URL : Document href from the document metadata
2. `https://api.sustainabilityreportingnavigator.com/api/documents/{doc_id}/download` : Provide the document  id in the url

Script folder : `srn_scrape`

### PDF parsing script

Install viper paser from [here](https://gitlab.cc-asp.fraunhofer.de/ppradhan/viper.git)

checkout branch - `viper-optimizations`

1. Set the file paths in viper_config.yaml
2. Run the python script : `python viper_parser.py --config /cluster/home/repo/my_llm_experiments/esrs_data_scraping/viper_config.yaml --cuda-ids 0 1 2 3 4 5 6 --num_workers 8`


### Database creation script:

1. set the path locations in main.yaml
2. Run the python script : `python main.py --config /cluster/home/repo/my_llm_experiments/esrs_data_scraping/main.yaml`
