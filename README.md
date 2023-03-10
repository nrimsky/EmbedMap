# Embedding Map Generator

Generate a 3D map of links based on their embeddings using OpenAI's embedding API

## How to use

- Clone this repo and cd into its root directory
- Create a file called `key.txt` in the root directory and paste in your OpenAI API key there
- Run `python3 -m venv venv` to create a virtual environment
- Run `source venv/bin/activate` to activate it
- Run `pip install -r requirements.txt` to install dependency
- Run `python main.py` to run the script (it will prompt you to provide URLs and page titles)
- You can view your completed map at `viz/index.html`. 
- You can drag the nodes of the map around to rotate the map in 3D
- You can also zoom in and out using normal zoom gestures
- Clicking on the nodes or text labels should take you to the original URLs

### Example
- You can see an example map at `example_viz/index.html` which was generated from some Wikipedia pages (the source data and embeddings can be found in `test_data` and `test_embeddings`)
- The `example_viz` folder is also publicly hosted so you can test interacting with the map at https://example-embed-map.netlify.app/ 