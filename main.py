import requests
from typing import List
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
import os
import json

URL = "https://api.openai.com/v1/embeddings"

with open("key.txt") as keyfile:
    API_KEY = keyfile.read().strip()


def get_embeddings(strings: List[str]) -> List[List[float]]:
    response = requests.post(URL, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }, json={
        "input": strings,
        "model": "text-embedding-ada-002"
    })
    try:
        return [response.json()["data"][i]["embedding"] for i in range(len(strings))]
    except KeyError:
        print("Error")
        print(response.text)
        exit()


def get_save_embeddings(files: List[tuple]):
    """
    files is list of (path of text file, path to save embeddings)
    """
    content = []
    orig_urls = []
    for filepath, _ in files:
        with open(filepath, "r") as txtfile:
            c = txtfile.read()
            if len(c) > 7000:
                content.append(c[:7000])
            else:
                content.append(c)
            orig_urls.append(c.split("\n")[0])
    embeddings = get_embeddings(content)
    proj = project_3d(embeddings)

    for i, (_, embedfilepath) in enumerate(files):
        with open(embedfilepath, "w") as embfile:
            embfile.write("URL\n")
            embfile.write(orig_urls[i])
            embfile.write("\nOriginal\n")
            embfile.write(str(list(embeddings[i])))
            embfile.write("\nProjected\n")
            embfile.write(str(list(proj[i])))


def get_data(url: str, save_to: str):
    """
    Save text from URL
    """
    txt = requests.get(url).text
    # Convert HTML into plain text
    soup = BeautifulSoup(txt, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    with open(save_to, "w") as txtfile:
        txtfile.write(url+"\n")  # Save URL data was fetched from as first line
        txtfile.write(text)


def get_wiki_page(name: str, data_dir="test_data"):
    """
    For fetching test data from Wikipedia
    """
    url = f"https://en.wikipedia.org/wiki/{name}"
    save_to = f"{data_dir}/{name}.txt"
    get_data(url, save_to)


def project_3d(embeddings: List[List[float]]) -> List[List[float]]:
    """
    n_examples x embed_dim -> n_examples x 3 (projected to 3 principal components)
    """
    pca = PCA(n_components=3)
    return pca.fit_transform(embeddings)


def run(data_dir="test_data", embedding_dir="test_embeddings", viz_dir="example_viz"):
    """
    Generate embeddings from text files and save to embeddings/ directory
    """
    data = [(f"{data_dir}/{n}", f"{embedding_dir}/{n}")
            for n in os.listdir(data_dir)]
    get_save_embeddings(data)
    make_viz(embedding_dir=embedding_dir, viz_dir=viz_dir)


def make_viz(embedding_dir="test_embeddings", viz_dir="example_viz"):
    """
    Create a JSON with all the embeddings mapped to the original filenames
    """
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    data = {"embeddings": []}
    for f in os.listdir(embedding_dir):
        with open(f"{embedding_dir}/"+f, "r") as ef:
            lines = ef.readlines()
            proj_embed = eval(lines[-1].strip())
            url = lines[1].strip()
            data["embeddings"].append({
                "link": url,
                "name": f.split(".")[0].replace("_", " "),
                "embedding": proj_embed
            })
    json_str = f"var data = {json.dumps(data)};\n"
    with open("index.html", "r") as htmlfile:
        html = htmlfile.read()
        html = html.replace("var data;", json_str)
        with open(f"{viz_dir}/index.html", "w") as outfile:
            outfile.write(html)


def gen_site_map(urls: List[str], titles: List[str], download_dir="data", embed_dir="embeddings", viz_dir="viz"):
    """
    urls: URLs to fetch data from
    titles: Names for these pages to appear as labels on Site Map
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    for url, title in zip(urls, titles):
        fname = title.replace(" ", "_")
        get_data(url, f"{download_dir}/{fname}.txt")
    run(data_dir=download_dir, embedding_dir=embed_dir, viz_dir=viz_dir)


def prompt_input():
    url = input("Enter URL (or press enter to stop adding URLs) >>> ")
    if len(url.strip()) == 0:
        return None, None
    title = input("Enter title (or press enter to stop adding URLs) >>> ")
    if len(title.strip()) == 0:
        return None, None
    return url.strip(), title.strip()


def main():
    urls = []
    titles = []
    while True:
        url, title = prompt_input()
        print("_____")
        if not url or not title:
            break
        urls.append(url)
        titles.append(title)
    if len(urls) > 0:
        print("Generating Map")
        gen_site_map(urls=urls, titles=titles)
        print("Done")


if __name__ == "__main__":
    main()
