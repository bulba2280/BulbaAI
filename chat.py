import torch
import torch.nn as nn
import random
import wikipediaapi
import feedparser
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from colorama import init, Fore, Style
import time
import sys


# цвета и анимация текста

init()

def color_print(text, color=Fore.CYAN, delay=0.03):
    
    for char in color + text + Style.RESET_ALL:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# настройки
sequence_length = 20
embed_size = 512
hidden_size = 1024
num_layers = 1              
dropout = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
color_print(f"Используем устройство: {device}", Fore.YELLOW, 0.02)

# сама модель
class MegaBot(nn.Module):
    def __init__(self, vocab_size, embed_size=embed_size, hidden_size=hidden_size,
                 num_layers=num_layers, dropout=dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc(self.dropout(last_output))
        return out

# загрузка ИИ
checkpoint = torch.load("bulbaAI.pth", map_location='cpu')
word2idx = checkpoint['word2idx']
idx2word = checkpoint['idx2word']
vocab_size = checkpoint['vocab_size']

model = MegaBot(vocab_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ответ
last_responses = []

def generate_response(text, max_len=2, temp=0.5):
    words = text.lower().split()
    idx = [word2idx.get(w, 0) for w in words][-sequence_length:]
    while len(idx) < sequence_length:
        idx.insert(0, 0)
    
    inp = torch.tensor([idx], device=device)
    
    res = []
    for _ in range(max_len):
        logits = model(inp)[0] / temp
        probs = torch.softmax(logits, 0)
        next_idx = torch.multinomial(probs, 1).item()
        word = idx2word[next_idx]
        
        if len(word) > 10:
            continue
            
        res.append(word)
        inp = torch.cat([inp[:, 1:], torch.tensor([[next_idx]], device=device)], 1)
    
    return ' '.join(res) if res else "да"

def answer_with_protection(text):
    global last_responses
    response = generate_response(text)
    
    bad_phrases = ["сама как", "что думаешь", "как считаешь", "как?", "а ты", "сама"]
    if any(bad in response.lower() for bad in bad_phrases):
        response = random.choice(["норм", "понял", "ок", "бывает", "ага", "да"])
    
    if response in last_responses[-2:]:
        response = random.choice(["ладно", "давай", "окей", "хорошо"])
    
    last_responses.append(response)
    if len(last_responses) > 3:
        last_responses.pop(0)
    
    return response

# вики
wiki = wikipediaapi.Wikipedia('BulbaAI/1.0', 'ru')

def wiki_search(query):
    page = wiki.page(query)
    if page.exists():
        summary = page.summary.split('.')[:1]
        return summary[0] + '.'
    else:
        return "Ничего не нашёл."

# новости
def get_news():
    feed = feedparser.parse('https://lenta.ru/rss')
    news = []
    for entry in feed.entries[:3]:
        title = entry.title.split('.')[0][:50]
        news.append(f"• {title}")
    return '\n'.join(news)


# гугл (в разработке)

def google_search(query, num_results=3):
    try:
        results = []
        for url in search(query, num_results=num_results, lang="ru"):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, timeout=3, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string.strip() if soup.title else "Без заголовка"
                results.append(f"• {title[:50]}...\n  {url}")
            except:
                results.append(f"• {url}")
        return '\n'.join(results) if results else "Ничего не нашёл."
    except Exception as e:
        return f"Ошибка поиска: {e}"


# сам чат

print("\n" + "*"*50)
color_print("BULBA AI", Fore.RED, 0.05)
print("$"*50)
color_print("Команды:", Fore.YELLOW, 0.03)
color_print("  /wiki текст — Википедия", Fore.CYAN, 0.02)
color_print("  /news — новости", Fore.CYAN, 0.02)
color_print("  /google текст — поиск в Google", Fore.CYAN, 0.02)
color_print("  exit — выход", Fore.CYAN, 0.02)
print("*"*50)

while True:
    user = input("\nТы: ").strip()
    
    if user.lower() == 'exit':
        color_print("Bulba: поки :3", Fore.MAGENTA, 0.04)
        break
    
    if user.startswith('/wiki '):
        query = user[6:]
        color_print("🔍 Википедия:", Fore.YELLOW, 0.02)
        color_print(wiki_search(query), Fore.GREEN, 0.02)
        continue
    
    if user.startswith('/news'):
        color_print("Новости:", Fore.YELLOW, 0.02)
        for news_line in get_news().split('\n'):
            color_print(news_line, Fore.CYAN, 0.02)
        continue
    
    if user.startswith('/google '):
        query = user[8:]
        color_print("Ищу в Google...", Fore.YELLOW, 0.02)
        color_print(google_search(query), Fore.MAGENTA, 0.02)
        continue
    
    response = answer_with_protection(user)
    color_print(f"BulbaAI: {response}", Fore.GREEN, 0.03)