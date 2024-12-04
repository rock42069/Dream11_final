'''
Schema:
[
  {
    "heading": heading,
    "description": description,
    "date": date,
    "link": path_on_website
  }, ..
]
'''
import requests, sys, json, datetime
from bs4 import BeautifulSoup
from multiprocessing import Pool, Lock, cpu_count

lock = Lock()
thread_count = cpu_count()

scheme = "https"
host = "www.espncricinfo.com"
heading_div_property = ['ds-text-title-s', 'ds-font-bold', 'ds-text-typo']

def getNewsData(link):
    while True:
        try:
            response = requests.get(f"{scheme}://{host}{link}")
            break
        except:
            print(f"Retrying => {scheme}://{host}{link}")
    html = BeautifulSoup(response.content, "html.parser")
    heading_tags = html.find_all("h2", attrs={"class": heading_div_property})
    heading_tags = heading_tags[:len(heading_tags)-5]
    info = [{"heading": heading_tag.text, "description": heading_tag.parent.find("p").text, "date": heading_tag.parent.find("span").text.split(' ')[0], "link": heading_tag.parent.parent.parent.parent.parent.get_attribute_list("href")[0]} for heading_tag in heading_tags]
    return info
def main(divided_data):
    current_data = []
    for link in divided_data:
        with lock:
            print(f"{scheme}://{host}{link}")
        current_data.extend(getNewsData(link))
    return current_data

if __name__ == "__main__":
    now = datetime.datetime.now()
    print(f"Starting Time => {now.strftime('%Y-%m-%d %H:%M:%S')}")
    total_pages = int(sys.argv[1])
    links = [f"/cricket-news?page={page_index}" for page_index in range(1, total_pages+1)]
    total_data = len(links)
    links_divisions = [links[index*total_data//thread_count: (index+1)*total_data//thread_count] for index in range(thread_count)]
    pool = Pool(thread_count)
    threads = []
    news_data = []
    for links_division in links_divisions:
        threads.append(pool.apply_async(main, (links_division, )))
    for thread in threads:
        news_data.extend(thread.get())
    pool.close()
    pool.join()
    with open("news.json", 'w') as file:
        json.dump(news_data, file)