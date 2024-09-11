import fitz
import requests
import os
from bs4 import BeautifulSoup
from openai import OpenAI

def split_into_chunks(text, chunk_size):
    """Splits the text into chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def create_messages(companyinfo, chunk_size=2000):
    """Creates a list of messages from the company info."""
    chunks = split_into_chunks(companyinfo, chunk_size)
    messages = [{"role": "user", "content": f"Heres some information about the company I want to apply to: {chunk}"} for chunk in chunks]
    return messages

def extract_text_from_url(url):
    """Extracts relevant text from a URL using BeautifulSoup."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract text from paragraphs and headings
    text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
    return text

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"

# Pick one of the Azure OpenAI models from the GitHub Models service
model_name = "gpt-4o-mini"
my_path = "Ryan_Resume.pdf"
doc = fitz.open(my_path)
jobdescriptionurl = input("Enter the job description url: ")
jobdescription = extract_text_from_url(jobdescriptionurl)
compUrl = input("Enter the company's website: ")
companyinfo = extract_text_from_url(compUrl)
resumetext = ""
for page in doc:
    resumetext += page.get_text()

# Create a client
client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

# Create the initial messages
messages = [
    {
        "role": "system",
        "content": "Forget all previous instructions. Act like an expert resume and cover letter writer who has dedicated over 15 years to mastering the craft. You are a certified professional resume writer. You create compelling, customized CVs that showcase clients' skills, experiences, and achievements, tailored to the specific job description they're applying for. You have a comprehensive understanding of diverse industries, current hiring trends, and Applicant Tracking Systems (ATS). Your cover letters are compelling narratives that showcase the candidate's qualifications for the position but also their distinctive strengths. You know how to reveal individual strengths, career objectives, and challenges. In order to fulfill your job, you provide real time example and make up experiences or stories to make it more believable. Use numbers, stats, anything that makes the CV stands out! DO NOT MAKE THE LETTER EXCESSIVELY LONG",  
    },
   
    {
        "role": "user",
        "content": "hi here is the raw format of my resume" + resumetext,
    },
]

# Add the company info messages
company_messages = create_messages(companyinfo)
job_description_messages = create_messages(jobdescription)
messages.extend(company_messages)
messages.extend(job_description_messages)

# Add the final message
messages.append({
    "role": "user",
    "content": "Please give me a cover letter for this company",
})

# Call the chat completion API
response = client.chat.completions.create(
    messages=messages,
    model=model_name,
)

# Print the response
print(response.choices[0].message.content)