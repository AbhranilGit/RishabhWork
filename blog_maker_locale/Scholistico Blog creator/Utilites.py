
from bs4 import BeautifulSoup
import os, io, openai, traceback, requests, time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import cloudinary
import cloudinary.uploader
from datetime import datetime
from tqdm.notebook import tqdm
import xml.etree.ElementTree as ET
from datetime import datetime
import re
import ipywidgets as widgets
from IPython.display import display, clear_output
from anthropic import Anthropic
import base64
from typing import List
import logging
import numpy as np
import anthropic
from anthropic import Anthropic
import pytesseract
import cv2
import uuid
import cloudinary
import cloudinary.uploader
import cloudinary.api
import re
import os
from PIL import Image
import io
from io import BytesIO
from PIL import Image
import re
from bs4 import BeautifulSoup
from slugify import slugify
import cloudinary
import cloudinary.uploader
import cloudinary.api
import re
import os
from PIL import Image
import json
import os
import base64
import json
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup
import shutil
import os
import re
import cloudinary
import cloudinary.uploader
import cloudinary.utils
import os
import hashlib
import time
import os
import logging
import pytesseract
import requests
import base64
from typing import List
import fal_client
import logging
import base64
import requests
from typing import List
import pytesseract
import openai

from bs4 import BeautifulSoup, Comment
client = Anthropic()

def create_formatted_html(input_html):
    soup = BeautifulSoup(input_html, 'html.parser')

    for tag in soup.find_all():
        if tag.name == 'h2':
            tag['class'] = tag.get('class', []) + ['wp-block-heading']
            tag.insert_before('<!-- wp:heading -->')
            tag.insert_after('<!-- /wp:heading -->')

        elif tag.name == 'h3':
            tag['class'] = tag.get('class', []) + ['wp-block-heading']
            tag.insert_before('<!-- wp:heading {"level":3} -->')
            tag.insert_after('<!-- /wp:heading -->')

        elif tag.name == 'p':
            tag.insert_before('<!-- wp:paragraph -->')
            tag.insert_after('<!-- /wp:paragraph -->')

        elif tag.name == 'ul':
            tag['class'] = tag.get('class', []) + ['wp-block-list']
            tag.insert_before('<!-- wp:list -->')
            tag.insert_after('<!-- /wp:list -->')

        elif tag.name == 'li':
            tag.insert_before('<!-- wp:list-item -->')
            tag.insert_after('<!-- /wp:list-item -->')

        elif tag.name == 'img':
            tag['class'] = tag.get('class', []) + ['wp-block-image', 'size-full']
            tag.insert_before('<!-- wp:image {"id":121096,"sizeSlug":"full","linkDestination":"media"} -->')
            tag.insert_after('<!-- /wp:image -->')

    # Add extra newlines for readability
    output = str(soup).replace('&lt;','<').replace('&gt;','>')
    output=output.replace('><','>\n\n<')
    output=output.replace('-->\n<!','-->\n\n<!')
    output=output.replace('>\n<!','>\n\n<!')
    output=output.replace('-->\n<','-->\n\n<')
    output=output.replace('>\n<','>\n\n<')

    return output.strip()

from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse

def extract_links(url):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'xml')
    links = {}

    # Define excluded extensions
    excluded_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.svg','.gif')

    for loc in soup.find_all('loc'):
        full_url = loc.text
        # Skip if URL ends with any of the excluded extensions
        if not full_url.lower().endswith(excluded_extensions):
            # Get the last part of the URL path as the key
            path = urlparse(full_url).path
            name = path.strip('/').split('/')[-1]
            if name:  # Only add if name is not empty
                links[name] = full_url

    return links

def change_to_working_directory():
    drive.mount('/content/drive')
    os.chdir("/content/drive/MyDrive/OldWorks/Rishabh/process/")

def get_topics_for_blog(course_name:str,course_link:str):
    return course_name,course_link

def read_xml_generate_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # result = generate_sitemap_dict(xml_content)

    root = ET.fromstring(xml_content)
    # Create a dictionary to store the results
    sitemap_dict = {}

    # Define the namespace
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    # Iterate through all <url> elements
    for url in root.findall('ns:url', namespace):
        # Get the <loc> element (URL)
        loc = url.find('ns:loc', namespace)
        if loc is not None:
            full_url = loc.text
            # Parse the URL to get the path
            parsed_url = urlparse(full_url)
            # Remove leading slash and .html extension if present
            path = parsed_url.path.strip('/').replace('.html', '')
            # Use the last part of the path as the key
            key = path.split('/')[-1]
            # Add to dictionary
            sitemap_dict[key] = full_url

    return sitemap_dict





def fetch_list_of_current_blogs():
    try:
        post_sitemap_url = 'https://de.scholistico.com/post-sitemap.xml'
        blog_links = extract_links(post_sitemap_url)
        return blog_links
    except Exception as e:

        return read_xml_generate_dict('XML_Sitemap_blog_Scholistico.xml')

def fetch_list_of_current_products():
    try:
        product_sitemap_url = 'https://de.scholistico.com/product-sitemap.xml'
        product_links = extract_links(product_sitemap_url)
        return product_links
    except Exception as e:
        return read_xml_generate_dict('XML_Sitemap_product_Scholistico.xml')

def get_client_selected_promotional_blog(blog_links:dict):
    last_three_items = dict(list(blog_links.items())[-3:])
    return last_three_items

def get_client_selected_promotional_product(product_links:dict):
    last_three_items = dict(list(product_links.items())[-3:])
    return last_three_items


def get_client_selected_inspirational_blog(blog_links:dict):
    items = dict(list(blog_links.items())[5:9])
    return items

def set_api_keys():
    os.environ["ANTHROPIC_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["PERPLEXITY_API_KEY"]=""
    os.environ["FAL_KEY"] = ""

def fetch_doc_ids(course_code):
    if course_code=='Art Therapy Practitioner':
        return  ['1CWjtMKQW-Oi2q7XOlyiO4aZe3flFVAL0DgpRdTiEcy0',
        '1ROFDBePG19f8Zajs8XbIHsyciZb_CyYRlV_UqFmwdNI',
        '1Sx8oM7xG8aulNjOfF-h2_hjGGPKOyNzp-GQO9wJTIwU',
        '1dwUI_MMddZXG9rkooURFNdTnuFE93AOUBjMT8ys8wSw',
        '1w8QGUQaAwrv56cbNX6LyDmI6rujX62IU7HA6cznJFUE',
        '1hg7u_7battYfXUk3UH2JwPaJtx3mrmObLHL-TxnRFqU',
        '1WHw4H-9RIlYBBOzLEbq5Vyl6WxfzA8sIOgCXrVpTv0M',
        '1Os0OXwCvQTmkLVJqod5x5N1Ob4lxZwXp831ljjRXLng',
        '1XRM8uOuNraV5_Uu13V9U6BHjB0SDP_87jw1j34wxtn4'
        ]



    if course_name=='Naturopathy Practitioner':
            return  ['1lqBptUgw9kFfQ3RkT1nTxbYxmWGpvwqpkOE1staPyN0',
            '1SztfRBYNYPLnnLybYG8s565bsgT5JPl5LP-xj_5bmK8',
            '1O2_jSS_5OrHIWlpzsDV_UIkllTx0swB1_Ae9RERVRrE',
            '1vKVQijYhO6YyRzxHwyMpolyLLyLpCpsrGgiON-6ItjA',
            '1XcgT2KBpZ34g0uoOnjjeI1vSnbNI2kDhTopFRMYHp0o',
            '1B1NyUhnXGnGpW9Ngr7G29zhUVQI6zcGJaPsgw8chURI',
            '1W9kCQosy5keiJGjQfcLZ04nTHxRITfsiPeSKdQ2Qec0',
            '1FBkqmXTxnqqUAwlvO42l_rcNQnOM8FzBoCjGt77rhVw',
            '1LKqBZktpWZGkRNklWdsG7IdQl7z5Phdf-hTKdm5B4Rg',
            '1IqO11VYPnO7JVpVWMfncpWXv8_lp_P-90WcQJHpqTQE',
            '10g0jsZhpm1r_GPNUongy10uNyJV4bHiKWfpj4VFvqhY',
            '1OYtiC-OGdzrhFGVssm8rg221RW5ZwDL0OupVMNDdXY4',
            '1ISU2Uxgwm2NyB9d4BFEQdmvvMorAAs8qjEBv6mwMCpY',
            '1Mnj9hlPhHVP7OGP5gQc_fq_SiFVV85bEUkABx4nJjfk',
            '18QLv_cjm75lul9E0kpoUBkgo5SsGrOeTc5N-v1jlQ6A']



    if course_name=='Holistic Health Practitioner':
            return  ['1HYxda0chv8hlLn1fweaqSNtj7ppQIzIxvn49-ftczIk',
            '1ClR70-4I5thG9JSB-9M55ZCLNP9NJcqJX7qEfWDG5pw',
            '1haFuGBOl_Kd_gj3iZdQdSK0z5hhrDi2ehTl2_BBw-iU',
            '17suZqERvACqJmEabcFBT6U31ZzQJp_WXvOVfGdWMGrA',
            '1jie5WD6yldXl4RWGfIhOOQmEg7aHwEenxCucncR2EwE',
            '17fTk-mRFpFvo3lWit5zRo56TA1JiTGI6FSNeVE4PeHI',
            '1E5Tzzp8hyDoKW6n01ec_h3zsU2sMdUH1IsBkP9kYC8Y',
            '1SNSrSR3H33H-goiB7nAFSRAbY01DWinI-tedvLWqzKE',
            '1toffafufyvJKLi3HDrcvoCe9HDxHw0Cgw1e-EHcLB3c',
            '1ku-0ehSAdq8rZmcC8fwk2b-eHBl8XiDt58DapKhN6vM',
            '1MMKfXwrfMYK53hZzOQF1CueTOcB8Nq_FdWlk6n9jMhU',
            '1j5asxdPAYFnmSjC6VQ09Y-PybL9qFewj7tRm05cA2dY']

    if course_name=='Sound Therapy Practitioner':
            return  ['18Vd2N2Zn_HbdM9s76-zPzjYRoNTedCtGu_mjNcfYzrY',
            '1TWQdV-6xcFcPJWezpNfBVfXkr7ugHy7KOCizAugI96c',
            '1K1d7bAAYG3EUkeKdlrKw4dCcZUkiUIJaNGCyiHuJxyc',
            '12jPCW08kRfUjwbA32eCl1aeiEHltEijjlAH2W1p8N1Q',
            '1I_ME9pFNDXlvgBCkRIt8rqjxHkdrExCg2iv-VU9RnAE',
            '1IDcvBdI5We-piZHFmaUcXOnAqs4jOwnLM1bt6o1YDeM',
            '16yH7H7ZLMOx_rpOKTlrd5NhJ3wyUJLGxUxkkTenYGmw']

    if course_name=='Animal Communication Specialist':
            return ['1F8B4YQa8_wHgFc6VXFwHAEK7m_Ik3S3c8v8GyQQWGhg',
                    '1hhUZuWcoQv3G_5E4DXvf2zY_S-LjYPY_gPCxBER3tQs',
                    '1RWYfkW76KHAfCkWh3VobtjTMk0qQkLiGuyPmKIN7YVU',
                    '1ymJWTr4f7XpfEGiCZUp8XdrR6OJQeDCh41cyBapxv5Q',
                    '1CsPnCip_aZ27HNdAzchPB9w2227nw9lPyBOk5rikkLo',
                    '1aXixZK1v0l6OwN5wBcDl1LGFaAmucb9TgzWEpoZsD6A',
                    '1JAIPh_aSiuIb3DvGBMPDGlpkzVQHJ6Pc8pGe3K1U3cU',
                    '1irtHDKpA9IDf1hz8TCHK20CezST9_hmVBMCMxMhhLFo',
                    '1a5SSZUuP242AyPKBZk1N2oZHoa62dch8_w5A2Shc2vY',
                    '1LeGmn9aEfRNv0ZP3gxmH-rs4Hlzydh7vyjANikof_LM']



def get_services():

    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    # Set up OpenAI API
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    auth.authenticate_user()

    # Build Google Drive and Docs services
    drive_service = build('drive', 'v3', cache_discovery=False)
    docs_service = build('docs', 'v1', cache_discovery=False)
    return client,openai,drive_service,docs_service

from docx import Document
import os

def read_module_intro(file_path):
    try:
        # Load the document
        doc = Document(file_path)
        
        # Extract text from the document
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'

        # Try to find "Module Introduction" first
        start_index = text.lower().find("module introduction")
        if start_index == -1:
            # If "Module Introduction" not found, try "Introduction"
            start_index = text.lower().find("introduction")
            if start_index == -1:
                raise ValueError("Could not find 'Module Introduction' or 'Introduction' in the document.")
            start = start_index + len("Introduction")
        else:
            start = start_index + len("Module Introduction")

        # Find the end index
        end_index = text.lower().find("module objectives", start_index)
        if end_index == -1:
            raise ValueError("Could not find 'Module Objectives' in the document.")

        return text[start:end_index].strip()

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def generate_module_summary(folder_path, output_file_path):
    """
    Generate summaries for module introductions from DOCX files in a folder.
    
    Args:
        folder_path (str): Path to the folder containing DOCX files
        output_file_path (str): Path where the output file will be saved
    """
    # Initialize API clients
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    def summarize_text(text):
        # Your existing summarize_text function implementation here
        # Make sure this function is defined or imported
        pass

    # Process each DOCX file in the folder
    with open(output_file_path, 'a', encoding='utf-8') as outfile:
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.docx'):
                file_path = os.path.join(folder_path, filename)
                try:
                    # Read module introduction from the DOCX file
                    module_intro = read_module_intro(file_path)
                    if module_intro:
                        # Generate summary
                        summary = summarize_text(module_intro)
                        # Write to output file
                        outfile.write(f"Summary from {filename}:\n{summary}\n\n")
                    else:
                        print(f"No module introduction found in {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


# Function to summarize text using OpenAI's GPT-4
def summarize_text(text):
    global client  # Use the global client instead of creating a new one

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following text in maximum 2 lines:\n{text}"
            }
        ]
    )

    return response.content[0].text.strip()



def generate_blog_topics(summaries, language):
    prompt = (
        f"You are a Content Writer in {language}"
        "Based on the following module summaries, please generate 8-10 blog post topics, section wise "
        "that would be interesting for potential customers of the course, without revealing "
        "too much course content. Make sure that each heading starts with a number example '7 fitness tips from Art Therapy'. The topics should focus on the impact of the course content "
        "on human well-being.\n\n"
        f"Module Summaries:\n{chr(10).join(summaries)}\n\n"
        "Blog Post Topics:"
    )

    client = Anthropic()

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=300,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the content from the response
    content = response.content[0].text

    # Split the content into individual blog topics
    blog_topics = content.strip().split('\n')
    while ("" in blog_topics):
        blog_topics.remove("")
    return blog_topics



def provide_blog_topics():

    output_file = 'process/module_summaries.txt'

    # Read the summaries from the output file
    with open(output_file, 'r') as file:
        module_summaries = file.read().splitlines()

    blog_topics = generate_blog_topics(module_summaries, language)

    while ("" in blog_topics):
        blog_topics.remove("")

    return blog_topics

def fetch_topic_for_blog(selected_blog_topic:str):
    return selected_blog_topic

def fetch_blog_specifications():
    num_of_words_in_blog= 2500
    num_of_images_in_blog= '7-8'
    num_of_headings= 7
    blog_percent= 95
    promotion_percent= 5
    num_of_infographics= 0
    user_notes_for_image= ""
    number_of_case_studies=3
    number_of_tables='1-2'
    num_of_outbound_links=4
    return num_of_words_in_blog,num_of_images_in_blog,num_of_headings,blog_percent,promotion_percent,num_of_infographics,user_notes_for_image,number_of_case_studies,number_of_tables,num_of_outbound_links



def optimize_prompt(prompt, context):
    # Construct the input for Claude
    input_text = f"""You are tasked with optimizing a prompt for DALLE image generation to create proper images for a blog post. Your goal is to refine the given prompt while considering the context of a previous image.

                    First, you will be provided with two inputs:
                    <prompt>{prompt}</prompt>
                    This is the initial prompt for DALLE image generation.

                    <content>"{context}"</content>
                    This represents the context from the previous image.

                    Your Key Focus should be of generating prompts for realistic images

                    To optimize the prompt for DALLE, follow these guidelines:

                    1. Analyze the given prompt and the previous image context.
                    2. Don't make the image too similar to the previous image.
                    3. Ensure the prompt is clear, specific, and descriptive.
                    4. Include relevant details from the previous image context to maintain consistency.
                    5. Use vivid and precise language to describe the desired image.
                    6. Incorporate artistic elements like style, mood, and composition if appropriate.
                    7. Avoid any potential copyright issues or explicit content.

                    Remember that DALLE has a token limit of 1000. To handle this:
                    - Keep your optimized prompt concise and within the token limit.
                    - Prioritize the most important elements of the image description.
                    - Remove any unnecessary words or repetitive information.

                    Provide your optimized prompt within <optimized_prompt> tags. After the optimized prompt, include a brief explanation of your changes and reasoning within <explanation> tags.

                    Your response should follow this format:

                    <optimized_prompt>
                    [Your optimized prompt here]
                    </optimized_prompt>

                    """

    # Call the Anthropic API
    # response = client.message.create(
    # model="claude-3-5-sonnet-20240620",
    # prompt=f"You are a prompt engineer who generates prompts for realistic image generation.\n\nUser: {input_text}",
    # max_tokens_to_sample=4000,
    # temperature=0.7
    # )

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4000,
        temperature=0.7,
        system="You are a prompt engineer who generates ONLY realistic Images. Images as close to real and human as possible",
        messages=[
            {
                "role": "user",
                "content": input_text
            }
        ]
    )

    # Extract the optimized prompt and explanation from the response
    full_response = response.content[0].text
    optimized_prompt = full_response.split('<optimized_prompt>')[1].split('</optimized_prompt>')[0].strip()
    # explanation = full_response.split('<explanation>')[1].split('</explanation>')[0].strip()

    return optimized_prompt

def generate_casestudy_prompts(client,topic: str, course_name: str, course_link: str, inspiration_links: List[str], number_of_prompts: int, language: str) -> List[str]:
    # Construct the prompt for Claude
    system_prompt = f"""You are an expert in creating educational content in {language}. Your task is to generate {number_of_prompts} prompts for real-world case studies related to the topic '{topic}' for the course '{course_name}' ({course_link}).

    Each case study prompt should:
    - Be based on actual events or companies
    - Provide specific details and outcomes
    - Illustrate key points discussed in the course
    - Relate to the surrounding content and enhance the reader's understanding
    - Accurately describe the relationship between the case study and the topic

    Use the following inspiration links for additional context:
    {', '.join(inspiration_links)}

    Please provide the prompts as a numbered list."""

    # Generate prompts using Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.7,
        system=system_prompt,
        messages=[
            {"role": "user", "content": "Generate the prompts as requested."}
        ]
    )

    # Extract the generated prompts from Claude's response
    generated_content = response.content[0].text
    prompts = [line.strip().split('. ', 1)[1] for line in generated_content.split('\n') if line.strip() and line[0].isdigit()]

    return prompts


def generate_case_study(prompt: str, language: str) -> Tuple[str, str]:
    # Get API key from environment variable
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set")

    # Set up the headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Set up the payload
    data = {
        "model": "llama-3.1-sonar-huge-128k-online",
        "messages": [
            {
                "role": "system",
                "content": f"You are a helpful assistant that generates case studies with working links in {language}."
            },
            {
                "role": "user",
                "content": f"Generate a case study based on the following prompt. Include at least one relevant, working link in your response: {prompt}"
            }
        ]
    }

    try:
        # Make the API call
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            json=data,
            headers=headers
        )
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the response
        result = response.json()
        case_study_text = result['choices'][0]['message']['content']

        # Extract the first link from the case study text
        links = re.findall(r'(https?://\S+)', case_study_text)
        case_study_link = links[0] if links else "No link found"

        return case_study_link, case_study_text

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the Perplexity API: {e}")
        return "", ""

def generate_outbound_link_prompts(client, topic: str, course_link: str, inspiration_links: List[str], number_of_prompts: int, language: str) -> List[str]:
    # Construct the prompt for Claude
    system_prompt = f"""You are an expert content curator in {language}. Your task is to generate {number_of_prompts} prompts for finding relevant outbound links related to the topic '{topic}' based on the course ({course_link}).

    Each prompt should:
    - Target high-quality, authoritative sources
    - Focus on specific aspects of the topic
    - Complement the course content
    - Not duplicate content from these inspiration links:
    {', '.join(inspiration_links)}

    Please provide the prompts as a numbered list."""

    # Generate prompts using Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.7,
        system=system_prompt,
        messages=[
            {"role": "user", "content": "Generate the prompts as requested."}
        ]
    )

    # Extract the generated prompts from Claude's response
    generated_content = response.content[0].text
    prompts = [line.strip().split('. ', 1)[1] for line in generated_content.split('\n') if line.strip() and line[0].isdigit()]

    return prompts

def generate_outbound_link(prompt: str, language: str) -> Tuple[str, str]:
    # Get API key from environment variable
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set")

    # Set up the headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Set up the payload
    data = {
        "model": "llama-3.1-sonar-huge-128k-online",
        "messages": [
            {
                "role": "system",
                "content": f"You are a helpful assistant that finds relevant, authoritative outbound links for blog posts in {language}."
            },
            {
                "role": "user",
                "content": f"Find a high-quality, relevant outbound link based on this prompt. Provide the link and a brief description of why it's relevant: {prompt}"
            }
        ]
    }

    try:
        # Make the API call
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            json=data,
            headers=headers
        )
        response.raise_for_status()

        # Parse the response
        result = response.json()
        link_suggestion = result['choices'][0]['message']['content']

        # Extract the link and description
        links = re.findall(r'(https?://\S+)', link_suggestion)
        outbound_link = links[0] if links else "No link found"

        return outbound_link, link_suggestion

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the Perplexity API: {e}")
        return "", ""

def get_outbound_links(client, topic: str, course_link: str, inspiration_links: List[str], number_of_links: int, language: str) -> List[dict]:
    # Generate prompts
    prompts = generate_outbound_link_prompts(
        client,
        topic=topic,
        course_link=course_link,
        inspiration_links=inspiration_links,
        number_of_prompts=number_of_links,
        language=language)

    # Generate links for each prompt
    outbound_links = []
    for prompt in prompts:
        link, description = generate_outbound_link(prompt,language)
        if link and link != "No link found":
            outbound_links.append(
                f"link: {link} , description: {description}"
            )

    return outbound_links

def generate_blog_post(topic: str, course_name: str, course_link: str,
                       list_of_blog_to_promote: dict, list_of_products_to_promote: dict,
                       num_of_words_in_blog: int, num_of_images_in_blog: int,
                       num_of_headings: int, blog_percent: int, promotion_percent: int,
                       num_of_infographics: int,number_of_casestudies:int, inspiration_links: List[str],number_of_tables,num_of_outbound_links:int, language: str) -> str:

    client = Anthropic()

    list_of_case_study_prompt=generate_casestudy_prompts(client,topic, course_name, course_link, inspiration_links, number_of_casestudies,language)

    case_studies="\n"
    case_study_links=[]
    for prompt in list_of_case_study_prompt:

        case_study_link, case_study_text = generate_case_study(prompt,language)
        case_study_links.append(case_study_link)
        case_studies+='\n CASE_STUDY_LINK \n: '+case_study_link+'\n\n\n\n'+'\n'+case_study_text
        case_studies+='\n\n\n\n'

    with open('casestudy_and_links.txt','w') as f:
        f.write(case_studies)

    def generate_links(items, class_name):
        return '\n'.join([f'<a href="{link}" class="{class_name}">{name}</a>' for name, link in items.items()])

    blog_links = generate_links(list_of_blog_to_promote, "blog-promo")
    product_links = generate_links(list_of_products_to_promote, "product-promo")

    def extract_main_content(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text

    def summarize_content(content, max_words=200):
        # Use Claude to summarize the content
        summary_prompt = f"Summarize the following content in about {max_words} words, focusing on the main ideas and structure:\n\n{content}"
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        return summary_response.content[0].text

    # Fetch and summarize content from inspiration links
    inspiration_summaries = []
    for link in inspiration_links:
        try:
            response = requests.get(link)
            if response.status_code == 200:
                main_content = extract_main_content(response.text)
                summary = summarize_content(main_content)
                inspiration_summaries.append(summary)
        except Exception as e:
            print(f"Error fetching content from {link}: {str(e)}")

    inspiration_text = "\n\n".join(inspiration_summaries)

    outbound_links=get_outbound_links(client, topic, course_link, inspiration_links, num_of_outbound_links, language)

    prompt = f"""You are tasked with creating a comprehensive blog post to promote a course while focusing primarily on a specific topic in {language}. Follow these instructions carefully to generate the required content:

    1. Topic and Course Information:
    <topic>{topic}</topic>
    <course_name>{course_name}</course_name>
    <course_link>{course_link}</course_link>

    2. Blog Post Structure and Requirements:
    a) Start with a heading (H1)
    b) Place an image signifying the topic
    c) Include a Infographic Table of Contents with links to headings. **Table of Contents is very important**
    d) Provide a topic description (promote the course here)
    e) Main content (promote products here)
    f) Conclusion

    - Write a blog post of {num_of_words_in_blog} words.
    - Include {num_of_images_in_blog} image placeholders with DALL-E prompts.
    - Create {num_of_headings} main headings and appropriate subheadings.
    - Incorporate {num_of_infographics} infographics encoded in HTML and CSS.
    - Focus {blog_percent}% on the topic and {promotion_percent}% on promoting the course, naturally intermixed.

    3. SEO Requirements:
    a) Include focus keyword in SEO Title
    b) Use focus keyword in SEO Meta Description
    c) Include focus keyword in URL
    d) Use focus keyword in the first 10% of the content
    e) Use focus keyword throughout the content

    4. Content Creation:
    a) Begin by analyzing the following inspiration content and extracting relevant ideas, structure, and tone:
    <inspiration>
    {inspiration_text}
    </inspiration>
    b) Based on the inspiration, outline the main sections of your blog post on the given topic.
    c) For each section, write detailed, informative content that provides value to the reader, incorporating ideas from the inspiration content where relevant.
    d) Naturally incorporate information about the course where relevant, ensuring it doesn't disrupt the flow of the main topic.
    e) Use clear, engaging language and maintain a consistent tone throughout the post, similar to the inspiration content.
    f) Seamlessly integrate mentions and links to all blogs and products from the provided lists throughout the content where relevant.
    g) Include case studies related to the topic. Each case study should:
      - Be based on actual events or companies
      - Provide specific details and outcomes
      - Illustrate key points discussed in the blog
      - display links of the case studies
    h)  Integrate all blogs from case_study_links naturally within the content without any CSS
        {case_study_links}
    i) Below are the details of above links' case studies
        {case_studies}
        ADDING CASE STUDIES IS A MUST TO THE BLOG POST.
    j) generate {number_of_tables} tables for the blog which fits in the blog very well.
    k-1) if the blog is about exercise then at the start of the explanation of exercise mention
      a) Time Required:   (example 30-60 mins)
      b) Materials Needed: (example crayon, yoga mat)
    k-2) at the end of the explanation of exercise mention
      a) Benefits: (example   - Research published in the Journal of Marital and Family Therapy suggests that collaborative art-making can improve couple communication and increase relationship satisfaction )
    note: for k-1) and k-2) keep Time Required:, Materials Needed:, Benefits: in Bold, also as you are a writer in {language} keep these headings also in {language}.
    l) keep a Section of \"References\" at the last of the Blog post. Displaying all the links for outbound case studies. No need to give description of the links.
    {outbound_links}
    m) keep below outbound links also present in the main content of the blog post.
    {outbound_links[0]}
    {outbound_links[1]}
    n) keep the language of the content formal in {language}

    5. HTML Structure:
    a) Start with <!DOCTYPE html> declaration
    b) Include proper <head> section with meta tags, title, and character encoding
    c) Use semantic HTML tags for headings (h1, h2, h3) and content organization
    d) Ensure proper nesting of all HTML elements
    e) Include viewport meta tag for responsiveness

    6. Image Placeholders and DALL-E Prompts:
    a) For each image placeholder, use the following format:
        <!-- DALL-E prompt: [detailed prompt here] -->
        <img src="PLACEHOLDER"/>
    b) Create detailed DALL-E prompts that relate to the surrounding content and enhance the reader's understanding.
    c) Image should be just after Headings
    d) Ensure the alt text accurately describes the intended image for accessibility.
    e) Don't make the image too similar to the previous image.



    7. Tables:
    a) Design {number_of_tables} Tables using HTML and CSS.
    b) Use <div> elements with appropriate classes for layout.
    c) Ensure the Tables are visually appealing and informative.

    8. Course and Product Promotion:
    a) Subtly mention the course throughout the post where relevant.
    b) only promote {course_name} once, if you have nothing to promote then don't promote.
    c) Use the following format for the course name, Include promotional links to the course page using the provided course_link, without any CSS:
        <span class="course-highlight"><a href="{course_link}" class="course-promo">{course_name}</a></span>Learn more about [aspect of topic]
        (only promote {course_name} once)
        NEVER PROMOTE {course_name} twice
    d) Integrate all blogs from list_of_blog_to_promote naturally within the content without any CSS:
    {blog_links}
    NEVER PROMOTE one blog twice.
    e) mention all the blogs to promote under section of \"Explore More\". As you are writer of {language} hence \"Explore More\" should be written in {language}.
    e) Integrate all products from list_of_products_to_promote seamlessly into the post without any CSS:
    {product_links}
    NEVER PROMOTE one product twice.

    9. Final Formatting and Review:
    a) Ensure all HTML tags are properly closed and nested.
    b) Verify that the word count is approximately {num_of_words_in_blog} words.
    c) Confirm that there are {num_of_images_in_blog} image placeholders with DALL-E prompts.
    d) Review the balance of topic content ({blog_percent}%) to course promotion ({promotion_percent}%).

    10. Real-World Examples and Data:
    a) When discussing key concepts or strategies, provide concrete examples from well-known companies or recent events.
    b) If applicable, mention any current trends or emerging technologies related to the topic.


    11. Expert Insights:
    a) If possible, include brief quotes or insights from industry experts or thought leaders in the field.
    b) Properly attribute any external quotes or paraphrased ideas.


    12. Conclusion:
    a) Write a proper conclusion that summarizes the main points of the blog post.
    b) Reinforce the value of the promoted course, blogs, and products in relation to the topic.
    c) Make sure that the post starts with <html>.
    d) Verify the HTML structure to be perfect before responding no unclosed tags, divs, DOM objects. [!!!! Very important]
    e) Remove any alt tag in the image placeholders with their closing tag '/>'.[!!!! Very important]

    13. Output Requirements:
    a) Start directly with <!DOCTYPE html> declaration
    b) Include complete HTML structure with head and body tags
    c) No introductory text or explanations before the HTML
    d) Ensure all tags are properly closed
    e) Use UTF-8 character encoding
    f) Include proper indentation for readability

    14.CONTENT GENERATION GUIDELINES:

      a) PROHIBITED WORDS/PHRASES:
      Maintain a complete ban on specified buzzwords and cliché phrases (like  DO NOT use any cases of these words or phrases in the content output including the title, body content, and sub-headings:
      "Encompass", "Captivate", "Dynamic", "Delve", "Testament", "Elevate", "Embark", "Enhance", "Embrace", "Crucial", "Propel", "Explore", "Leverage", "Synergize", "Fast-paced", "Comprehensive", "Harness",
      "Spearhead", "Unlock", "Energize", "Navigate", "Seamless", "Cultivate", "Buckle up", "Delve into", "Ever-changing", "Evolving", "Ever-evolving", "Blooming", "Dive into", "Diving into", "Daunting",
      "Unleash", "Mastermind", "Master", "Whiz", "Grasp", "Realm", "Landscape", "Remember", "It's important to note", "In conclusion", "In summary", "Remember that", "Furthermore", "However",
      "First things first", "Second", "First", "Therefore", "Additionally", "In the dynamic world of", "Embark on a journey", "In this digital landscape", "A treasure trove of",
      "In this digital world", "In the fast-paced world", "Testament to", "Gone are the days", "Look no further", "Dreamt of".)
      Avoid transition phrases and common content fillers
      Skip overused digital marketing terminology


      b) WRITING MECHANICS:
      Use everyday language and clear expressions
      Keep sentences varied in length (mix short and medium)
      Maintain active voice (80% minimum)
      Break down complex ideas into simple explanations
      Avoid technical jargon unless absolutely necessary
      If using technical terms, provide simple explanations


      c) PERSONAL TOUCH:
      Include first-person perspectives ("I," "me," "my")
      Share relevant personal experiences
      Create direct dialogue with readers
      Add authentic anecdotes when relevant


      d) CONTENT TONE:
        i) Academic Focus:
        Clear learning objectives
        Logical information flow
        Real examples and case studies
        Mix of theory and practice
        Well-organized sections with clear headings

        ii) Voice and Approach:
        Friendly but professional
        Educational without being patronizing
        Conversational yet informative
        Inclusive language ("we," "you")
        Practical and solution-focused

        iii) Presentation:
        Break information into digestible chunks
        Balance historical context with current relevance
        Include both advantages and limitations
        Connect concepts to real-world applications
        Focus on reader benefit and practical value

      The output should feel like an experienced professional having an informed conversation with a peer, while maintaining educational value and credibility.


    15. Technical Requirements:
        a) Use proper HTML5 semantic elements
        b) Include necessary meta tags
        c) Ensure all links have proper href attributes
        d) Use proper HTML entities for special characters
        e) Maintain consistent indentation
        f) Include proper language attribute in html tag

        Remember:
        - Start directly with HTML code
        - No introductory text
        - No explanatory comments before or after the HTML
        - Ensure complete and valid HTML structure
        - Maintain proper tag closure and nesting

    16. HTML Validation Requirements:
        a) Ensure DOCTYPE declaration is in uppercase: <!DOCTYPE html>
        b) Include lang attribute in html tag: <html lang="{language}">
        c) Mandatory meta tags in head:
            - <meta charset="UTF-8">
            - <meta name="viewport" content="width=device-width, initial-scale=1.0">
            - <meta name="description" content="[SEO description]">
        d) All void elements must end with />
        e) All attributes must use double quotes
        f) Proper nesting hierarchy must be maintained
        g) No inline styles - all CSS in <style> tag
        h) All IDs must be unique
        i) All images must have alt attributes

    17. HTML Structure Validation:
        a) Required structure:
            <!DOCTYPE html>
            <html lang="{language}">
            <head>
                [meta tags]
                <title>[title]</title>
                <style>[styles]</style>
            </head>
            <body>
                [content]
            </body>
            </html>
        b) No content outside these tags
        c) Proper indentation (4 spaces)
        d) No empty elements without closing tags
        e) Consistent quotation mark usage

    18. Content Container Requirements:
        a) Main content must be wrapped in <main> tag
        b) Sections must use <section> tags
        c) Articles must use <article> tags
        d) Navigation must use <nav> tags
        e) Footer must use <footer> tag
        f) Proper use of <aside> for supplementary content

    19. Error Prevention:
        a) No unclosed tags
        b) No mismatched tags
        c) No improper nesting
        d) No duplicate IDs
        e) No missing required attributes
        f) No content before DOCTYPE
        g) No content after closing html tag

    20. Output Validation Steps:
        a) Verify complete HTML structure
        b) Check all tags are properly closed
        c) Validate nested elements
        d) Confirm proper attribute syntax
        e) Ensure semantic HTML usage
        f) Verify character encoding declaration

    Remember to integrate the case studies, articles, examples, and expert insights naturally throughout the blog post, ensuring they enhance the overall narrative and provide additional value to the reader. Use actual, working URLs for all external links.

    don't mention anything like "Here's the completed blog post as requested:" just pure html blog post.
    blog post should start with html Header not text.

    NO PREAMBLE`


    """
    # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  
    # openai.api_key = os.environ.get("OPENAI_API_KEY")
    previous_html = ""
    while True:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        # response = openai.ChatCompletion.create(
        #     model="chatgpt-4o-latest",  # Replace with the correct model name
        #     messages=[
        #         {"role": "system", "content": f"You are a blog creator in {language} who generates blog in {num_of_words_in_blog} words"},
        #         {"role": "user", "content": prompt}
        #     ],
        #     max_tokens=4096,  # Adjust as needed for your use case
        #     temperature=0.7
        # )

        # response = openai.chat.completions.create(
        #     model="chatgpt-4o-latest",
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=100000,
        #     temperature=0.7
        # )

        # new_content = message.choices[0].message.content
        # generated_html = new_content
        
        generated_html = message.content[0].text
        # If there's previous HTML and current HTML ends with </html>, combine them
        if previous_html and generated_html.strip().endswith("</html>"):
            generated_html = previous_html + generated_html

        # Save the generated HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attempt_{timestamp}.html"
        filepath = os.path.join("process/attempts", filename)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(generated_html)

        # Check if the HTML is complete
        if generated_html.strip().endswith("</html>"):
            print("Generated HTML ends with </html>")
            return generated_html
        else:
            previous_html = generated_html
            prompt = f"""
            previous prompt was :{prompt}
            {generated_html}
            The HTML above is incomplete. Please complete it, ensuring it ends with </html>.


            don't start anything with Here's the completed HTML, ending with

            just pure html blog post.
            """
            continue

    return generated_html

def add_content_id(html_content):
    # Function to generate a unique content ID
    def generate_content_id():
        return f"content-{uuid.uuid4().hex[:8]}"

    # Regular expression to match HTML tags
    tag_pattern = r'<([a-zA-Z][a-zA-Z0-9]*)(\s[^>]*)?>'

    # Function to add content ID to matched tags
    def add_id_to_tag(match):
        tag_name = match.group(1)
        attributes = match.group(2) or ''
        content_id = generate_content_id()

        # Add content_id attribute right after the tag name
        return f'<{tag_name} content_id="{content_id}"{attributes}>'

    # Use re.sub to replace all matched tags with tags containing content ID
    modified_html = re.sub(tag_pattern, add_id_to_tag, html_content)

    return modified_html

# def extract_dalle_prompts(blog_content: str) -> List[str]:
#     return re.findall(r'<!-- DALL-E prompt: (.*?) -->', blog_content, re.DOTALL)

def extract_dalle_prompts(blog_content: str) -> List[str]:
    # Pattern to match comments containing DALL-E prompts, now with content_id
    pattern = r'<!--\s*DALL-E prompt:\s*(.*?)\s*-->'

    # Find all matches
    matches = re.findall(pattern, blog_content, re.DOTALL | re.IGNORECASE)

    # Strip whitespace from each prompt
    prompts = [prompt.strip() for prompt in matches]

    return prompts


import fal_client
import logging
import base64
import requests
from typing import List
import cv2
import pytesseract

def generate_images(prompts: List[str], user_notes_for_image: str) -> List[str]:
    images = []

    # Fetch previous_context from the specified file
    try:
        img = cv2.imread('process/context/img1.png')
        if img is None:
            raise FileNotFoundError("Image file not found")

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use pytesseract to extract text from the image
        previous_context = pytesseract.image_to_string(gray)
    except Exception as e:
        logging.error(f"Error reading previous context: {str(e)}")
        previous_context = ""

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(log["message"])

    for prompt in prompts:
        full_prompt = optimize_prompt(prompt, previous_context)
        full_prompt += f"""
    Include:
    - Key elements
    - Environment
    - Emotion
    - Lighting

    Style:
    - High-quality photography
    - Natural background
    - Full frame

    Note:
    {user_notes_for_image}
    - Positive, ethical theme
    - Visual storytelling
    - Realism, no text
    """

        try:
            result = fal_client.subscribe(
                "fal-ai/flux-realism",
                arguments={
                    "prompt": f"""Here is the user topic. "{full_prompt}".
                                  If you find any inappropriate content in the prompt, please ignore it and don't generate inappropriate images""",
                    "image_size": "landscape_16_9",  # Changed to a valid option
                    "negative_prompt": "fingers, blurry, low quality, low resolution, pixelated, artifacts",
                },
                with_logs=True,
                on_queue_update=on_queue_update,
            )

            # Extract the image URL from the result
            image_url = result['images'][0]['url']
            image_data = base64.b64encode(requests.get(image_url).content).decode('utf-8')
            images.append(image_data)

            # Update previous context
            previous_context = prompt

            # Log additional information
            logging.info(f"Image generated. Width: {result['images'][0]['width']}, Height: {result['images'][0]['height']}")
            logging.info(f"Content Type: {result['images'][0]['content_type']}")
            logging.info(f"Seed: {result['seed']}")
            logging.info(f"NSFW concepts detected: {result['has_nsfw_concepts'][0]}")

        except Exception as e:
            logging.error(f"Error generating image for prompt: {prompt}. Error: {str(e)}")
            raise Exception(f"Error generating image for prompt: {prompt}. Error: {str(e)}")

            images.append("")

    return images



def load_image_paths(json_path: str) -> Dict[str, str]:
    with open(json_path, 'r') as f:
        return json.load(f)

def save_image_paths(image_path_dict: Dict[str, str], json_path: str):
    with open(json_path, 'w') as f:
        json.dump(image_path_dict, f, indent=2)
    print(f"Saved image paths to: {json_path}")

def save_images(image_dict: Dict[str, str], save_path: str, title_dict: Dict[str,str]) -> Dict[str, str]:
    image_path_dict = {}
    os.makedirs(save_path, exist_ok=True)
    for content_id, image_base64 in image_dict.items():
        try:
            image_data = base64.b64decode(image_base64)
            file_path = os.path.join(save_path, f"{title_dict[content_id]}.png")
            image_path_dict[content_id] = file_path
            with open(file_path, "wb") as f:
                f.write(image_data)
            print(f"Saved image: {file_path}")
        except Exception as e:
            print(f"Error saving image for {content_id}: {str(e)}")
    return image_path_dict

def create_image_dict(html_content: str, images: list, titles: list) -> tuple[dict, dict]:
    soup = BeautifulSoup(html_content, 'html.parser')
    image_tags = soup.find_all('img', attrs={'content_id': True, 'src': 'PLACEHOLDER'})
    image_dict = {img['content_id']: image for img, image in zip(image_tags, images)}
    title_dict = {img['content_id']: title for img, title in zip(image_tags, titles)}
    
    return title_dict,image_dict

def update_html_with_images(html_content: str, json_path: str,website: str) -> str:
    # Load image paths from JSON file
    image_path_dict = load_image_paths(json_path)

    soup = BeautifulSoup(html_content, 'html.parser')
    image_tags = soup.find_all('img', attrs={'content_id': True, 'src': 'PLACEHOLDER'})
    # Fetch current month and year
    current_month = str(datetime.now().month).zfill(2)
    current_year = datetime.now().year

    for img_tag in image_tags:
        content_id = img_tag['content_id']
        if content_id in image_path_dict:
            image_path = image_path_dict[content_id]
            title=image_path.split('/')[-1].split('.')[0]
            try:
                with open(image_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                # img_tag['src'] = f"data:image/png;base64,{img_data}"
                img_tag['src'] = f'https://{website}.com/wp-content/uploads/{current_year}/{current_month}/{title}.jpg'
            except FileNotFoundError:
                print(f"Warning: Image file not found: {image_path}")

    return str(soup)



def save_html(html_content,base_path,identifier=None):
    # Generate identifier based on content and timestamp
    if identifier==None:
        identifier = generate_identifier(html_content)

    # Ensure the base path exists
    os.makedirs(base_path, exist_ok=True)

    # Create the full file path
    file_path = os.path.join(base_path, f"{identifier}.html")

    # Save the HTML content
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(html_content)

    return identifier

def generate_identifier(content: str) -> str:
    # Combine content with current timestamp
    timestamp = str(time.time())
    combined = content + timestamp

    # Generate a hash of the combined string
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()

    # Use the first 8 characters of the hash as the identifier
    return hash_hex[:8]

def extract_numbered_sentences(text_list):
    numbered_sentences = []
    for text in text_list:
        if re.match(r'^\d+\.', text):
            numbered_sentences.append(text)
    return numbered_sentences

def get_blog_topic_from_client(blog_topics):
    return blog_topics[1]

import json
import os
import anthropic

def generate_titles_for_prompts(prompts, client):
    
    prompt_titles = []
    
    for prompt in prompts:
        ai_prompt = f"""
        Generate a short, file-friendly title (max 100 characters) for an image with this prompt:
        
        "{prompt}"
        
        Requirements:
        - Maximum 100 characters
        - No spaces (use underscores if needed)
        - Only alphanumeric characters and underscores
        - Should be descriptive of the image content
        - NO PREAMBLE, just return the title
        """

        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0.7,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": ai_prompt
                        }
                    ]
                }
            ]
        )

        title = message.content[0].text.strip()
        

        title = ''.join(c for c in title if c.isalnum() or c == '_')  # Remove invalid chars
        
        prompt_titles.append(title)

    return prompt_titles



def add_styling(html_content: str) -> str:
    css = """
          <style>
          body {
              font-family: Arial, sans-serif;
              line-height: 1.6;
              max-width: 800px;
              margin: 0 auto;
              padding: 20px;
              background-color: #f9f9f9;
              color: #313234;
          }

          h1 {
              color: #3f4f19;
              font-family: "Merriweather", serif;
          }

          h2 {
              color: #33411e;
              font-family: "Merriweather", serif;
          }

          h3, h4, h5, h6 {
              color: #33411e;
              font-family: "Merriweather", serif;
          }

          a {
              color: #495238;
              text-decoration: none;
          }

          a:hover {
              color: #495238;
              text-decoration: underline;
          }

          .entry-content {
              letter-spacing: -0.24px;
          }

          .entry-content p, .entry-content li {
              font-size: 18px;
              line-height: 1.6;
          }


          img {
              max-width: 100%;
              height: auto;
              display: block;
              margin: 20px 0;
              border-radius: 5px;
          }

          .infographic {
              background-color: #2B5600;
              padding: 20px;
              border-radius: 5px;
              margin: 20px 0;
              color: #495238;
          }

          .button {
              background-color: #364915;
              color: #FFFFFF;
              border: 1px solid #212303;
              padding: 10px 20px;
              text-decoration: none;
              display: inline-block;
              border-radius: 3px;
          }

          .button:hover {
              background-color: #2C7000;
          }

          .secondary-button {
              background-color: #2B5600;
              color: #495238;
              border: 1px solid #2B5600;
              padding: 10px 20px;
              text-decoration: none;
              display: inline-block;
              border-radius: 3px;
          }

          .secondary-button:hover {
              background-color: #2B5600;
              color: #495238;
              border-color: #2B5600;
          }
          </style>
    """
    return f"{css}\n{html_content}"



def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            # Get the width and height
            width, height = img.size
            return width, height
    except IOError:
        print(f"Unable to open image file: {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None







def generate_slug_and_meta(html_content,client):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the title
    title = soup.title.string if soup.title else ''

    # Generate slug from the title
    slug = slugify(title)

    # Extract all text from the body
    body_text = soup.body.get_text() if soup.body else ''

    # Clean up the text
    body_text = re.sub(r'\s+', ' ', body_text).strip()

    # # Use Claude to generate meta description
    # client = anthropic.Client(api_key)
    prompt = f"Given the following webpage content, generate a concise and engaging meta description of about 150-160 characters. NO PREAMBLE:\n\nTitle: {title}\n\nContent: {body_text[:1000]}..."

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    meta_description = message.content[0].text



    return {
        'slug': slug,
        'meta_description': meta_description
    }
def GenerateBlogStructure(topic,
                          course_name,
                          course_link,
                          results_blog_dict,
                          results_product_dict,
                          num_of_words_in_blog,
                          num_of_images_in_blog,
                          num_of_headings,
                          blog_percent,
                          promotion_percent,
                          num_of_infographics,
                          number_of_case_studies,
                          inpiration_blog_dict,
                          number_of_tables,
                          num_of_outbound_links,
                          language
                          ):
    blog_content = generate_blog_post(topic,
                                      course_name,
                                      course_link,
                                      results_blog_dict,
                                      results_product_dict,
                                      num_of_words_in_blog,
                                      num_of_images_in_blog,
                                      num_of_headings,
                                      blog_percent,
                                      promotion_percent,
                                      num_of_infographics,
                                      number_of_case_studies,
                                      inpiration_blog_dict.values(),
                                      number_of_tables,
                                      num_of_outbound_links,
                                      language
                                      )


    blog_content=add_content_id(blog_content)

    structure_path='Blogs/Organization/html/blog_structure/'



    blog_identifier=save_html(blog_content,structure_path)


    return blog_identifier

def SaveImages(blog_identifier):
    structure_path='Blogs/Organization/html/blog_structure/'
    with open(f"{structure_path}{blog_identifier}.html") as f:
          blog_content=f.read()

    prompts = extract_dalle_prompts(blog_content)
    images= generate_images(prompts, user_notes_for_image)

    image_titles=generate_titles_for_prompts(prompts,client)

    image_path=f'Blogs/Organization/images/{blog_identifier}/'
    json_path=f'Blogs/Organization/json/{blog_identifier}.json'

    title_dict,image_dict = create_image_dict(blog_content, images, image_titles)
    image_path_dict = save_images(image_dict, image_path,title_dict)
    save_image_paths(image_path_dict, json_path)

def ProduceFinalOutput(blog_identifier):
    json_path=f'Blogs/Organization/json/{blog_identifier}.json'
    image_title_json=f'Blogs/Organization/json/title_{blog_identifier}.json'
    structure_path='Blogs/Organization/html/blog_structure/'
    with open(f"{structure_path}{blog_identifier}.html") as f:
          blog_content=f.read()
    updated_html = update_html_with_images(blog_content, json_path)
    print("Add styling")
    # final_html = add_styling(updated_html)
    # final_html=updated_html.replace('<img','<figure')
    updated_html=create_formatted_html(updated_html)
    output_path='Blogs/Organization/html/final_output/'
    save_html(updated_html,output_path,blog_identifier)

def DeleteImage(blog_identifier, Content_id):
    image_path = f'Blogs/Organization/images/{blog_identifier}/{Content_id}.png'

    width = height = None  # Initialize width and height

    try:
        if os.path.exists(image_path):
            # Get image size before deleting
            with Image.open(image_path) as img:
                width, height = img.size

            os.remove(image_path)
            print(f"Image deleted successfully: {image_path}")
        else:
            print(f"Image not found: {image_path}")
    except Exception as e:
        print(f"Error deleting image: {e}")

    return width, height


cloudinary.config(
    cloud_name = "ddzaqiihn",
    api_key = "626813739546286",
    api_secret = "MXztcr6KdWsM7XnYGLdBeXegnIw"
)


def AddImage(source_path, blog_identifier, Content_id, width=None, height=None):
    destination_path = f'Blogs/Organization/images/{blog_identifier}/{Content_id}.png'

    try:
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Upload the image to Cloudinary
        upload_result = cloudinary.uploader.upload(source_path)

        # Generate the Cloudinary URL with transformations
        transformation = []
        if width and height:
            transformation = [
                {'width': width, 'height': height, 'crop': 'fill'}
            ]

        resized_url = cloudinary.utils.cloudinary_url(
            upload_result['public_id'],
            transformation=transformation
        )[0]

        # Download the image from Cloudinary
        response = requests.get(resized_url)
        if response.status_code == 200:
            # Save the image content to the destination path
            with open(destination_path, 'wb') as f:
                f.write(response.content)

            print(f"Image added successfully: {destination_path}")
            return resized_url
        else:
            print(f"Failed to download image from Cloudinary. Status code: {response.status_code}")

    except FileNotFoundError:
        print(f"Source file not found: {source_path}")
    except PermissionError:
        print(f"Permission denied. Unable to upload the file.")
    except Exception as e:
        print(f"Error adding image: {e}")

    return None


def UpdateBlogStructure(html_identifier, content_id, updated_text):
    html_file_path=f'Blogs/Organization/html/blog_structure/{html_identifier}.html'
    # Read HTML from the identifier (file path)
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the element with the specified content_id
    element = soup.find(content_id=content_id)

    if element:
        # Update the text content of the element
        element.string = updated_text

        # Write the updated HTML back to the file
        with open(html_file_path, 'w', encoding='utf-8') as file:
            file.write(str(soup))

        print(f"HTML updated successfully. Content with ID '{content_id}' has been updated.")
    else:
        print(f"Element with ID '{content_id}' not found in the HTML.")

def create_simple_overlay(image_path, output_path, title, subtitle,
                          margin_ratio=0.15,
                          box_fill_color=(93, 108, 50, 153),
                          border_color="white",
                          border_thickness=2,
                          inner_margin_ratio=0.05):


    # Load the image
    background = Image.open(image_path).convert("RGBA")
    img_width, img_height = background.size

    # Calculate the margins
    margin_x = int(img_width * margin_ratio)
    margin_y = int(img_height * margin_ratio)

    # Calculate square size based on available height
    square_size = img_height - (2 * margin_y)

    # Coordinates for the square box
    left = margin_x
    top = margin_y
    right = left + square_size
    bottom = top + square_size

    # Calculate inner border coordinates (5% inside)
    border_offset = int(square_size * inner_margin_ratio)
    inner_left = left + border_offset
    inner_top = top + border_offset
    inner_right = right - border_offset
    inner_bottom = bottom - border_offset

    # Draw the overlay
    overlay = Image.new("RGBA", background.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw main box
    draw.rectangle([left, top, right, bottom], fill=box_fill_color)

    # Draw inner border
    draw.rectangle([inner_left, inner_top, inner_right, inner_bottom],
                  outline=border_color, width=border_thickness)

    # Set initial font size
    font_path = "Merriweather-Bold.ttf"
    title_font_size = 20
    subtitle_font_size = 20
    title_font = ImageFont.truetype(font_path, title_font_size)
    subtitle_font = ImageFont.truetype(font_path, subtitle_font_size)

    # Adjust the text area to fit within the inner border
    text_area_width = inner_right - inner_left
    text_area_height = inner_bottom - inner_top

    # Dynamically adjust the title font size for the new text area
    while True:
        title_width, title_height = get_multiline_text_size(title, title_font, draw)
        if title_width <= text_area_width * 0.9 and title_height <= text_area_height * 0.6:
            title_font_size += 2
            title_font = ImageFont.truetype(font_path, title_font_size)
        else:
            title_font_size -= 2
            title_font = ImageFont.truetype(font_path, title_font_size)
            break

    # Dynamically adjust the subtitle font size
    while True:
        subtitle_width, subtitle_height = get_multiline_text_size(subtitle, subtitle_font, draw)
        if subtitle_width <= square_size * 0.9 and subtitle_height <= square_size * 0.3:
            subtitle_font_size += 2
            subtitle_font = ImageFont.truetype(font_path, subtitle_font_size)
        else:
            subtitle_font_size -= 2
            subtitle_font = ImageFont.truetype(font_path, subtitle_font_size)
            break
    # Center text within the square
    title_width, title_height = get_multiline_text_size(title, title_font, draw)
    subtitle_width, subtitle_height = get_multiline_text_size(subtitle, subtitle_font, draw)

    # Define spacing between title and subtitle (adjust this value as needed)
    title_subtitle_spacing = square_size * 0.1  # 10% of square size for spacing

    # Adjust font sizes
    # Make subtitle smaller relative to title
    while True:
        subtitle_width, subtitle_height = get_multiline_text_size(subtitle, subtitle_font, draw)
        if subtitle_width <= square_size * 0.8:  # Making subtitle width smaller than title
            break
        subtitle_font_size -= 2
        subtitle_font = ImageFont.truetype(font_path, subtitle_font_size)

    # Ensure subtitle is proportionally smaller than title
    if subtitle_font_size > title_font_size * 0.6:  # Subtitle will be 60% of title size
        subtitle_font_size = int(title_font_size * 0.6)
        subtitle_font = ImageFont.truetype(font_path, subtitle_font_size)
        subtitle_width, subtitle_height = get_multiline_text_size(subtitle, subtitle_font, draw)

    # Calculate text positions with new spacing
    total_content_height = title_height + title_subtitle_spacing + subtitle_height
    start_y = top + (square_size - total_content_height) // 2

    # Position for title
    title_x = left + (square_size - title_width) // 2
    title_y = start_y

    # Position for subtitle
    subtitle_x = left + (square_size - subtitle_width) // 2
    subtitle_y = title_y + title_height + title_subtitle_spacing

    # Draw the text
    draw.multiline_text(
        (title_x, title_y),
        title,
        font=title_font,
        fill="white",
        align="center"
    )

    draw.multiline_text(
        (subtitle_x, subtitle_y),
        subtitle,
        font=subtitle_font,
        fill="white",
        align="center"
    )

    # Merge overlay with background and save
    final_image = Image.alpha_composite(background, overlay)
    final_image.save(output_path)


