"""Daily RSS -> original article text -> grounded Korean briefing -> GitHub Pages JSON.
Run with GEMINI_API_KEY; no model-generated content is published without a source URL."""
import os, json, re, hashlib, datetime as dt, time, logging
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
import requests, feedparser, trafilatura
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'
TODAY=dt.datetime.now(ZoneInfo('Asia/Seoul')).date()
MODEL=os.getenv('GEMINI_MODEL','gemini-2.5-flash')
MAX_PER_CATEGORY=int(os.getenv('MAX_PER_CATEGORY','5'))
# Direct publisher RSS feeds. Update feeds when providers change their URLs.
FEEDS={
 '경영·경제·금융':[
  'https://www.theguardian.com/business/rss',
  'https://feeds.bbci.co.uk/news/business/rss.xml',
  'https://www.theguardian.com/business/economics/rss'],
 '국내 정치·사회·정책':[
  'https://www.koreatimes.co.kr/www/rss/nation.xml',
  'https://www.koreaherald.com/rss/020100000000.xml',
  'https://www.hani.co.kr/rss/'],
 '국제 정세·외교·안보':[
  'https://feeds.bbci.co.uk/news/world/rss.xml',
  'https://www.theguardian.com/world/rss'],
 '과학·기술·환경':[
  'https://feeds.bbci.co.uk/news/technology/rss.xml',
  'https://www.theguardian.com/technology/rss',
  'https://www.theguardian.com/environment/rss',
  'https://feeds.bbci.co.uk/news/science_and_environment/rss.xml']}
UA={'User-Agent':'PersonalNewsReader/1.0 (+RSS aggregation; contact repository owner)'}
client=genai.Client(api_key=os.environ["GEMINI_API_KEY"])
logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')

def read_index():
 try:return json.loads((DATA/'index.json').read_text('utf-8'))
 except (OSError,ValueError):return {'dates':[]}

def previous_titles():
 titles=[]
 for date in read_index()['dates'][:5]:
  try:titles.extend(x['title'] for x in json.loads((DATA/f'{date}.json').read_text('utf-8'))['articles'])
  except (OSError,ValueError,KeyError):pass
 return titles[-100:]

def collect(category):
 results=[]; seen=set()
 for feed_url in FEEDS[category]:
  try:
   r=requests.get(feed_url,timeout=16,headers=UA);r.raise_for_status()
   feed=feedparser.parse(r.content)
   for item in feed.entries[:25]:
    url=item.get('link','')
    if not url.startswith('http') or url in seen:continue
    seen.add(url)
    published=item.get('published_parsed') or item.get('updated_parsed')
    if not published:continue # no date -> cannot verify freshness
    date=dt.date(*published[:3]);age=(TODAY-date).days
    if age< -1 or age>3:continue
    summary=BeautifulSoup(item.get('summary',''),'html.parser').get_text(' ',strip=True)
    results.append({'url':url,'title':item.get('title',''),'snippet':summary[:450],
                    'published':date.isoformat(),'source':urlparse(url).hostname or ''})
  except Exception as exc:logging.warning('Feed failed %s: %s',feed_url,exc)
 return results[:70]

def select(category,candidates,old_titles):
 if not candidates:return []
 instructions='''당신은 한국어 뉴스 편집자입니다. 제공된 RSS 후보만 평가하세요. 다른 기사를 만들지 마세요.
중요한 공적 이슈, 구체적 새 사실, 해당 분야 적합성, 서로 다른 주제의 다양성을 우선하세요.
연예, 스포츠, 가십, 광고, 의견만 있는 글, 동일 사건 중복은 제외하세요.
이전 기사 제목과 같은 사건이면 의미 있는 후속 변화가 드러날 때만 포함하세요.
국내 분야는 대한민국의 국회·정책·법원·교육·노동·복지·사회현안 위주이며 한국 이외 지역 기사는 제외하세요.
응답은 JSON 객체 {"indices":[정수,...]} 하나만, 후보의 0부터 시작하는 인덱스를 최대 10개 순서대로 반환하세요.'''
 prompt=json.dumps({'category':category,'previous_titles':old_titles[:70], 'candidates':[
  {'i':i,'title':c['title'],'snippet':c['snippet'][:170],'date':c['published']} for i,c in enumerate(candidates)]},ensure_ascii=False)
 res=client.models.generate_content(
 model=MODEL,
 contents=prompt,
 config=types.GenerateContentConfig(
  system_instruction=instructions,
  response_mime_type="application/json",
  max_output_tokens=1000
 )
)
indices=json.loads(res.text).get('indices',[])indices=json.loads(res.output_text).get('indices',[])
 return [candidates[i] for i in indices if isinstance(i,int) and 0<=i<len(candidates)][:10]

def fetch_article(url):
 try:
  r=requests.get(url,timeout=20,headers=UA,allow_redirects=True);r.raise_for_status()
  if 'text/html' not in r.headers.get('Content-Type','text/html'):return None
  txt=trafilatura.extract(r.text,include_comments=False,include_tables=False,favor_precision=True)
  if not txt or len(txt)<650:return None
  return txt[:18000]
 except Exception as exc:logging.warning('Article unavailable %s: %s',url,exc);return None

def summarize(category,article,body):
 instructions='''당신은 사실 검증을 중시하는 한국어 뉴스 해설자입니다. 독자는 중학생 수준의 용어 설명이 필요하지만 심도 있는 기사 내용을 원합니다.
제공한 기사 본문에 명시된 정보만 사실로 쓰고, 외부 지식을 지어내지 마세요. 사실과 전망을 구분하세요.
summary: 기사 핵심을 5~9문장으로 충분히 상세히 서술하세요. 숫자, 사건, 주요 관계자 발언과 반대 입장을 가능하면 포함하세요.
background: 기사에 나온 관련 인물의 역할, 과거 사건 및 이해에 필요한 배경을 3~5문장으로 서술하되 기사에 없는 구체적 과거 사실을 창작하지 마세요.
impact: 단순한 중요성 설명 대신 인과관계, 이해관계자별 영향, 반대 시나리오, 불확실성을 4~6문장으로 분석하세요. 추론은 '가능성'으로 표시하세요.
terms: 전문용어 1~3개를 쉽고 정확하게 풀이하세요.
허위정보·선정성·편향을 피하세요. 기사의 내용이 너무 빈약하면 모든 필드를 비워 응답하세요.
반드시 JSON 객체 {"title":"...","summary":"...","background":"...","impact":"...","terms":"..."}만 반환하세요.'''
 payload=json.dumps({'category':category,'article_title':article['title'],'date':article['published'],
 'source':article['source'],'original_text':body},ensure_ascii=False)
 response=client.models.generate_content(
 model=MODEL,
 contents=payload,
 config=types.GenerateContentConfig(
  system_instruction=instructions,
  response_mime_type="application/json",
  max_output_tokens=4096
 )
)
return json.loads(response.text) 
return json.loads(response.output_text)

def main():
 if not os.getenv('GEMINI_API_KEY'):raise RuntimeError('GEMINI_API_KEY is required.')
 DATA.mkdir(exist_ok=True); index=read_index(); results=[]; prior=previous_titles(); urls=set()
 for category in FEEDS:
  logging.info('Collecting %s',category)
  choices=select(category,collect(category),prior)
  count=0
  for item in choices:
   if count>=MAX_PER_CATEGORY:break
   if item['url'] in urls:continue
   body=fetch_article(item['url'])
   if not body:continue
   try:
    info=summarize(category,item,body)
    if not all(len(info.get(k,'').strip())>=25 for k in ('title','summary','background','impact')):continue
    uid=hashlib.sha256(item['url'].encode()).hexdigest()[:16]
    results.append({'id':uid,'category':category,'url':item['url'],'source':item['source'],
                    'published':item['published'],**{k:info.get(k,'') for k in ('title','summary','background','impact','terms')}})
    urls.add(item['url']);count+=1
    logging.info('Added %s',item['title'][:75])
   except Exception as exc:logging.warning('Summarization failed: %s',exc)
   time.sleep(.5)
 if not results:
  logging.warning('No verified articles today. Preserving previous briefing and index.');return
 filename=DATA/f'{TODAY.isoformat()}.json'
 document={'date':TODAY.isoformat(),'generated_at':dt.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST'), 'articles':results}
 filename.write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
 index['dates']=sorted(set(index['dates']+[TODAY.isoformat()]),reverse=True)[:90]
 (DATA/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
 logging.info('Published %d stories for %s',len(results),TODAY)
if __name__=='__main__':main()
