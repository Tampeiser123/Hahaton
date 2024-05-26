from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

def ensure_scheme(url):
    if not urlparse(url).scheme:
        return 'https://' + url
    return url

def fetch_page_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def extract_meta_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    title = soup.title.string if soup.title else ''
    meta_description = ''
    meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
    if meta_desc_tag:
        meta_description = meta_desc_tag.get('content', '')
        
    h1_tags = [h1.get_text() for h1 in soup.find_all('h1')]
    
    return {
        'title': title,
        'meta_description': meta_description,
        'h1_tags': h1_tags,
        'content': soup.get_text()
    }

def remove_stop_words(text):
    words = text.split()
    return ' '.join([word for word in words if word.lower() not in stop_words])

def analyze_similarity(pages):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([page['content'] for page in pages])
    cosine_similarities = cosine_similarity(tfidf_matrix)
    
    return cosine_similarities

def check_robots_txt(url):
    url = ensure_scheme(url)
    robots_url = urljoin(url, 'robots.txt')
    response = requests.get(robots_url)
    if response.status_code == 200:
        return 'robots.txt найден', 10
    else:
        return 'robots.txt не найден', -10

def check_duplicate_content(url, pages):
    url = ensure_scheme(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    duplicate_content_score = 0
    for page in pages:
        if page != url:
            if text == page['content']:
                duplicate_content_score -= 10
                break
    return 'Дублирующийся контент найден' if duplicate_content_score < 0 else 'Дублирующийся контент не найден', duplicate_content_score

def check_code_errors(url):
    url = ensure_scheme(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    errors = soup.find_all(text=lambda text: isinstance(text, str) and 'error' in text.lower())
    return 'Ошибки кода найдены' if errors else 'Ошибки кода не найдены', -10 if errors else 10

def check_internal_links(url):
    url = ensure_scheme(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    internal_links = 0
    domain = urlparse(url).netloc
    for link in soup.find_all('a', href=True):
        if domain in link['href']:
            internal_links += 1
    return f'Внутренних ссылок: {internal_links}', 10 if internal_links > 5 else -10

def check_html_errors(url):
    url = ensure_scheme(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    errors = soup.find_all(text=lambda text: isinstance(text, str) and 'error' in text.lower())
    if errors:
        return 'Ошибки найдены', -10
    else:
        return 'Ошибки не найдены', 10

def check_links(url):
    url = ensure_scheme(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=True)
    if len(links) > 5:
        return f'Найдено {len(links)} ссылок', 10
    else:
        return f'Найдено {len(links)} ссылок', -10

def check_image_weights(url):
    url = ensure_scheme(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    images = soup.find_all('img')
    total_weight = 0
    for img in images:
        img_url = urljoin(url, img['src'])
        img_response = requests.head(img_url)
        if 'Content-Length' in img_response.headers:
            total_weight += int(img_response.headers['Content-Length'])
    if total_weight / 1024 < 1024:
        return f'Общий вес картинок: {total_weight / 1024:.2f} KB', 10
    else:
        return f'Общий вес картинок: {total_weight / 1024:.2f} KB', -10

def check_page_speed(url):
    url = ensure_scheme(url)
    start_time = time.time()
    response = requests.get(url)
    load_time = time.time() - start_time
    if load_time < 2:
        return f'Скорость загрузки страницы: {load_time:.2f} секунд', 10
    else:
        return f'Скорость загрузки страницы: {load_time:.2f} секунд', -10

def check_meta_tags(url):
    url = ensure_scheme(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    meta_tags = soup.find_all('meta')
    meta_info = {tag.get('name', ''): tag.get('content', '') for tag in meta_tags if tag.get('name')}
    if 'description' in meta_info and 'keywords' in meta_info:
        return meta_info, 10
    else:
        return meta_info, -10

def check_text(url):
    url = ensure_scheme(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    if text:
        return text[:200], 10  # Первые 200 символов текста
    else:
        return 'Текст не найден', -10

def check_video(url):
    url = ensure_scheme(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    videos = soup.find_all('video')
    if videos:
        return f'Найдено {len(videos)} видео элементов', 10
    else:
        return f'Видео элементы не найдены', 0  # Нейтральный балл

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    urls = request.form.get('urls').split()
    urls = [ensure_scheme(url) for url in urls if url]

    pages = []

    for url in urls:
        html_content = fetch_page_content(url)
        if html_content:
            meta_data = extract_meta_data(html_content)
            meta_data['content'] = remove_stop_words(meta_data['content'])
            pages.append(meta_data)
    
    similarities = analyze_similarity(pages)
    
    for i in range(len(urls)):
        for j in range(i+1, len(urls)):
            print(f"Similarity between {urls[i]} and {urls[j]}: {similarities[i][j]:.2f}")

    results = []

    for url in urls:
        page_result = {}
        page_result['url'] = url
        
        robots_txt_result, robots_txt_score = check_robots_txt(url)
        duplicate_content_result, duplicate_content_score = check_duplicate_content(url, pages)
        code_errors_result, code_errors_score = check_code_errors(url)
        html_errors_result, html_errors_score = check_html_errors(url)
        internal_links_result, internal_links_score = check_internal_links(url)
        links_result, links_score = check_links(url)
        image_weights_result, image_weights_score = check_image_weights(url)
        page_speed_result, page_speed_score = check_page_speed(url)
        meta_tags_result, meta_tags_score = check_meta_tags(url)
        text_result, text_score = check_text(url)
        video_result, video_score = check_video(url)
        
        page_result['robots_txt'] = robots_txt_result
        page_result['duplicate_content'] = duplicate_content_result
        page_result['code_errors'] = code_errors_result
        page_result['html_errors'] = html_errors_result
        page_result['internal_links'] = internal_links_result
        page_result['links'] = links_result
        page_result['image_weights'] = image_weights_result
        page_result['page_speed'] = page_speed_result
        page_result['meta_tags'] = meta_tags_result
        page_result['text'] = text_result
        page_result['video'] = video_result

        results.append(page_result)
    
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
