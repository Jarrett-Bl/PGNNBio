import re
import requests

from bs4 import BeautifulSoup

"""
Crawl the MSigDB website to get the pathway ID for each pathway URL.
"""
def crawl_MSigDB(url: str):
    response = requests.get(url)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    title = soup.find('title').text
    
    if title == 'GSEA | MSigDB | Gene Set Not Found':
        return 'None'
    
    table = soup.find('table', {'class': 'lists4 human'})  
    th = table.find('th', text='Exact source')
    next_tag = th.find_next_sibling()
    pathway_id = next_tag.text
      
    return pathway_id

def fill_in_ids(file_name: str):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        
    for i, line in enumerate(lines):
        match = re.search(r'http://.*?(?=\t)', line) 
        if match:
            url = match.group(0)
            pathway_id = crawl_MSigDB(url)
            lines[i] = line.replace('\t', f'\t{pathway_id}\t', 1)
        
    with open(file_name, 'w') as file:
        file.writelines(lines)
            
            